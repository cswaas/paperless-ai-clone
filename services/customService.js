const {
  calculateTokens,
  calculateTotalPromptTokens,
  truncateToTokenLimit,
  writePromptToFile
} = require('./serviceUtils');
const OpenAI = require('openai');
const config = require('../config/config');
const tiktoken = require('tiktoken');
const paperlessService = require('./paperlessService');
const fs = require('fs').promises;
const path = require('path');
const RestrictionPromptService = require('./restrictionPromptService');

// Best-effort JSON cleaner to salvage slightly invalid model responses.
function cleanAndParseJson(rawContent) {
  const originalContent = String(rawContent || '');
  let jsonContent = originalContent;
  const repairNotes = [];

  const markReplace = (input, regex, replacement, note) => {
    const next = input.replace(regex, replacement);
    if (next !== input && note) {
      repairNotes.push(note);
    }
    return next;
  };

  // Prefer fenced JSON blocks if present
  const fencedMatch = jsonContent.match(/```json([\s\S]*?)```/i);
  if (fencedMatch) {
    jsonContent = fencedMatch[1];
  }

  // Strip code fences
  jsonContent = jsonContent.replace(/```json\s*/gi, '').replace(/```/g, '').trim();

  // Take substring from first '{'
  const firstBrace = jsonContent.indexOf('{');
  if (firstBrace !== -1) {
    jsonContent = jsonContent.slice(firstBrace);
  }

  // Fix missing commas between array object entries: `}{` -> `},{`
  jsonContent = jsonContent.replace(/}\s*{\s*"/g, '},{"');

  // Close unclosed value strings that break at newline
  jsonContent = jsonContent.replace(/("value"\s*:\s*")([^"\n]*)(\n|$)/g, '$1$2"$3');

  // Avoid aggressively collapsing braces here; let bracket balancing handle missing closures

  // Close unterminated value strings that run into a closing array/brace without quotes/commas
  jsonContent = markReplace(
    jsonContent,
    /(\"value\"\s*:\s*\"[^"\]\n]*)(\s*\])/g,
    '$1"}]',
    'closed-value-before-array'
  );

  const balanceBrackets = (input) => {
    let output = input.replace(/,\s*([}\]])/g, '$1');

    // Track bracket order so we append closers in the correct sequence (e.g. ] before })
    const stack = [];
    let inString = false;
    let escapeNext = false;

    for (let i = 0; i < output.length; i++) {
      const ch = output[i];

      if (escapeNext) {
        escapeNext = false;
        continue;
      }

      if (ch === '\\') {
        escapeNext = true;
        continue;
      }

      if (ch === '"') {
        inString = !inString;
        continue;
      }

      if (inString) continue;

      if (ch === '{' || ch === '[') {
        stack.push(ch);
      } else if (ch === '}' || ch === ']') {
        // pop the matching opener if present
        if (stack.length && ((ch === '}' && stack[stack.length - 1] === '{') || (ch === ']' && stack[stack.length - 1] === '['))) {
          stack.pop();
        }
      }
    }

    // Append missing closers in reverse order of openings
    while (stack.length) {
      const opener = stack.pop();
      output += opener === '{' ? '}' : ']';
    }

    return output;
  };

  const tryParse = (input) => JSON.parse(input);

  const balancedContent = balanceBrackets(jsonContent);
  const normalizeTail = (input) => input
    // Remove stray trailing commas before array/obj closers
    .replace(/,\s*\]/g, ']')
    .replace(/,\s*\}/g, '}');
  const normalizedContent = normalizeTail(balancedContent);
  try {
    const parsed = tryParse(normalizedContent);
    return { parsed, repaired: repairNotes.length > 0, cleaned: normalizedContent, repairNotes };
  } catch (firstError) {
    let repairedContent = jsonContent;

    // Close lines with odd quote counts (common when a value string is left open)
    repairedContent = repairedContent
      .split('\n')
      .map(line => {
        const quoteCount = (line.match(/"/g) || []).length;
        if (quoteCount % 2 === 1) return `${line}"`;

        // If value line ends without closing quote, add it
        if (/"value"\s*:\s*"[^"\n]*$/.test(line.trim())) {
          return `${line}"`;
        }

        return line;
      })
      .join('\n');

    repairedContent = balanceBrackets(repairedContent);
    const normalizedRepaired = normalizeTail(repairedContent);

    try {
      return { parsed: tryParse(normalizedRepaired), repaired: true, cleaned: normalizedRepaired, firstError, repairNotes };
    } catch (secondError) {
      const error = new Error(`Failed to parse JSON after repair: ${secondError.message}`);
      error.firstError = firstError;
      error.secondError = secondError;
      error.cleaned = normalizedContent;
      error.repaired = normalizedRepaired;
      error.original = originalContent;
      throw error;
    }
  }
}

class CustomOpenAIService {
  constructor() {
    this.client = null;
    this.tokenizer = null;
  }

  initialize() {
    if (!this.client && config.aiProvider === 'custom') {
      this.client = new OpenAI({
        baseURL: config.custom.apiUrl,
        apiKey: config.custom.apiKey
      });
    }
  }

  async analyzeDocument(content, existingTags = [], existingCorrespondentList = [], existingDocumentTypesList = [], id, customPrompt = null, options = {}) {
    const cachePath = path.join('./public/images', `${id}.png`);
    try {
      this.initialize();
      const now = new Date();
      const timestamp = now.toLocaleString('de-DE', { dateStyle: 'short', timeStyle: 'short' });

      if (!this.client) {
        throw new Error('Custom OpenAI client not initialized');
      }

      // Handle thumbnail caching
      try {
        await fs.access(cachePath);
        console.log('[DEBUG] Thumbnail already cached');
      } catch (err) {
        console.log('Thumbnail not cached, fetching from Paperless');

        const thumbnailData = await paperlessService.getThumbnailImage(id);

        if (!thumbnailData) {
          console.warn('Thumbnail nicht gefunden');
        }

        await fs.mkdir(path.dirname(cachePath), { recursive: true });
        await fs.writeFile(cachePath, thumbnailData);
      }

      // Format existing tags
      let existingTagsList = existingTags.join(', ');

      // Get external API data if available and validate it
      let externalApiData = options.externalApiData || null;
      let validatedExternalApiData = null;

      if (externalApiData) {
        try {
          validatedExternalApiData = await this._validateAndTruncateExternalApiData(externalApiData);
          console.log('[DEBUG] External API data validated and included');
        } catch (error) {
          console.warn('[WARNING] External API data validation failed:', error.message);
          validatedExternalApiData = null;
        }
      }

      let systemPrompt = '';
      let promptTags = '';
      const model = config.custom.model;

      // Parse CUSTOM_FIELDS from environment variable
      let customFieldsObj;
      try {
        customFieldsObj = JSON.parse(process.env.CUSTOM_FIELDS);
      } catch (error) {
        console.error('Failed to parse CUSTOM_FIELDS:', error);
        customFieldsObj = { custom_fields: [] };
      }

      // Generate custom fields template for the prompt
      const customFieldsTemplate = {};

      customFieldsObj.custom_fields.forEach((field, index) => {
        customFieldsTemplate[index] = {
          field_name: field.value,
          value: "Fill in the value based on your analysis"
        };
      });

      // Convert template to string for replacement and wrap in custom_fields
      const customFieldsStr = '"custom_fields": ' + JSON.stringify(customFieldsTemplate, null, 2)
        .split('\n')
        .map(line => '    ' + line)  // Add proper indentation
        .join('\n');

      // Get system prompt based on configuration
      if (config.useExistingData === 'yes' && config.restrictToExistingTags === 'no' && config.restrictToExistingCorrespondents === 'no') {
        systemPrompt = `
        Pre-existing tags: ${existingTagsList}\n\n
        Pre-existing correspondents: ${existingCorrespondentList}\n\n
        Pre-existing document types: ${existingDocumentTypesList.join(', ')}\n\n
        ` + process.env.SYSTEM_PROMPT + '\n\n' + config.mustHavePrompt.replace('%CUSTOMFIELDS%', customFieldsStr);
        promptTags = '';
      } else {
        config.mustHavePrompt = config.mustHavePrompt.replace('%CUSTOMFIELDS%', customFieldsStr);
        systemPrompt = process.env.SYSTEM_PROMPT + '\n\n' + config.mustHavePrompt;
        promptTags = '';
      }

      // Process placeholder replacements in system prompt
      systemPrompt = RestrictionPromptService.processRestrictionsInPrompt(
        systemPrompt,
        existingTags,
        existingCorrespondentList,
        existingDocumentTypesList,
        config
      );

      // Include validated external API data if available
      if (validatedExternalApiData) {
        systemPrompt += `\n\nAdditional context from external API:\n${validatedExternalApiData}`;
      }

      if (process.env.USE_PROMPT_TAGS === 'yes') {
        promptTags = process.env.PROMPT_TAGS;
        systemPrompt = `
        Take these tags and try to match one or more to the document content.\n\n
        ` + config.specialPromptPreDefinedTags;
      }

      // Custom prompt override if provided
      if (customPrompt) {
        console.log('[DEBUG] Replace system prompt with custom prompt');
        systemPrompt = customPrompt + '\n\n' + config.mustHavePrompt;
      }

      // Calculate tokens AFTER all prompt modifications are complete
      const totalPromptTokens = await calculateTotalPromptTokens(
        systemPrompt,
        process.env.USE_PROMPT_TAGS === 'yes' ? [promptTags] : [],
        model
      );

      const maxTokens = Number(config.tokenLimit);
      const reservedTokens = totalPromptTokens + Number(config.responseTokens);
      const availableTokens = maxTokens - reservedTokens;

      // Validate that we have positive available tokens
      if (availableTokens <= 0) {
        console.warn(`[WARNING] No available tokens for content. Reserved: ${reservedTokens}, Max: ${maxTokens}`);
        throw new Error('Token limit exceeded: prompt too large for available token limit');
      }

      console.log(`[DEBUG] Token calculation - Prompt: ${totalPromptTokens}, Reserved: ${reservedTokens}, Available: ${availableTokens}`);
      console.log(`[DEBUG] Use existing data: ${config.useExistingData}, Restrictions applied based on useExistingData setting`);
      console.log(`[DEBUG] External API data: ${validatedExternalApiData ? 'included' : 'none'}`);

      const truncatedContent = await truncateToTokenLimit(content, availableTokens, model);

      // console.log('######################################################################');
      // console.log(`[DEBUG] Content length: ${content.length}, Truncated content length: ${truncatedContent.length}`);
      // console.log(`[DEBUG] Truncated content: ${truncatedContent}`);
      // console.log(`[DEBUG] System prompt: ${systemPrompt}`);
      // console.log(`[DEBUG] Prompt tags: ${promptTags}`);
      // console.log(`[DEBUG] Model: ${model}`);
      // console.log(`[DEBUG] Custom fields: ${customFieldsStr}`);
      // console.log(`[DEBUG] Existing tags: ${existingTagsList}`);
      // console.log(`[DEBUG] Existing correspondents: ${existingCorrespondentList}`);
      // console.log(`[DEBUG] Custom prompt: ${customPrompt}`);
      // console.log(`[DEBUG] External API data: ${validatedExternalApiData}`);
      // console.log('######################################################################');


      const response = await this.client.chat.completions.create({
        model: model,
        messages: [
          {
            role: "system",
            content: systemPrompt
          },
          {
            role: "user",
            content: truncatedContent
          }
        ],
        response_format: { type: 'json_object' },
        temperature: 0.3,
      });

      // Handle response
      //console.log(`MESSAGE: ${response?.choices?.[0]?.message?.content}`);
      if (!response?.choices?.[0]?.message?.content) {
        throw new Error('Invalid API response structure');
      }

      // Log token usage
      console.log(`[DEBUG] [${timestamp}] Custom OpenAI request sent`);
      if (response.usage?.total_tokens !== undefined) {
        console.log(`[DEBUG] [${timestamp}] Total tokens: ${response.usage.total_tokens}`);
      } else {
        console.warn('[WARNING] Custom provider did not return usage metrics');
      }

      const usage = response.usage;
      const mappedUsage = usage ? {
        promptTokens: usage.prompt_tokens,
        completionTokens: usage.completion_tokens,
        totalTokens: usage.total_tokens
      } : null;

      let jsonContent = response.choices[0].message.content;
      console.log('[DEBUG] Raw provider content (first 500 chars):', String(jsonContent).slice(0, 500));
      const logDir = path.join(__dirname, '..', 'logs');
      await fs.mkdir(logDir, { recursive: true });
      await fs.appendFile(path.join(logDir, 'response-raw.txt'), `doc:${id ?? 'n/a'}\n${String(jsonContent)}\n---\n`);

      let parsedResponse;
      try {
        const parseResult = cleanAndParseJson(jsonContent);
        parsedResponse = parseResult.parsed;
        await fs.appendFile(path.join(logDir, 'response.txt'), `${JSON.stringify(parsedResponse)}\n`);

        if (parseResult.repaired) {
          console.warn(`[WARNING] Response required auto-repair${id ? ` for doc ${id}` : ''}`);
          const repairNoteStr = (parseResult.repairNotes && parseResult.repairNotes.length)
            ? `notes:${parseResult.repairNotes.join(',')}`
            : 'notes:none';
          await fs.appendFile(path.join(logDir, 'response-repaired.txt'), `doc:${id ?? 'n/a'}\n${repairNoteStr}\n${parseResult.cleaned}\n---\n`);
        }
      } catch (error) {
        await fs.appendFile(path.join(logDir, 'response-error.txt'), `doc:${id ?? 'n/a'}\n${String(jsonContent)}\n---\n`);
        console.error(`Failed to parse JSON response for doc ${id}:`, error, '\nRaw content:', jsonContent);
        throw new Error(`Invalid JSON response from API for doc ${id}`);
      }

      // Validate response structure
      if (!parsedResponse || !Array.isArray(parsedResponse.tags) || typeof parsedResponse.correspondent !== 'string') {
        throw new Error('Invalid response structure: missing tags array or correspondent string');
      }

      return {
        document: parsedResponse,
        metrics: mappedUsage,
        truncated: truncatedContent.length < content.length
      };
    } catch (error) {
      console.error('Failed to analyze document:', error);
      return {
        document: { tags: [], correspondent: null },
        metrics: null,
        error: error.message
      };
    }
  }

  /**
   * Validate and truncate external API data to prevent token overflow
   * @param {any} apiData - The external API data to validate
   * @param {number} maxTokens - Maximum tokens allowed for external data (default: 500)
   * @returns {string} - Validated and potentially truncated data string
   */
  async _validateAndTruncateExternalApiData(apiData, maxTokens = 500) {
    if (!apiData) {
      return null;
    }

    const dataString = typeof apiData === 'object'
      ? JSON.stringify(apiData, null, 2)
      : String(apiData);

    // Calculate tokens for the data
    const dataTokens = await calculateTokens(dataString, config.custom.model);

    if (dataTokens > maxTokens) {
      console.warn(`[WARNING] External API data (${dataTokens} tokens) exceeds limit (${maxTokens}), truncating`);
      return await truncateToTokenLimit(dataString, maxTokens, config.custom.model);
    }

    console.log(`[DEBUG] External API data validated: ${dataTokens} tokens`);
    return dataString;
  }

  async analyzePlayground(content, prompt) {
    const musthavePrompt = `
    Return the result EXCLUSIVELY as a JSON object. The Tags and Title MUST be in the language that is used in the document.:  
        {
          "title": "xxxxx",
          "correspondent": "xxxxxxxx",
          "tags": ["Tag1", "Tag2", "Tag3", "Tag4"],
          "document_date": "YYYY-MM-DD",
          "language": "en/de/es/..."
        }`;

    try {
      this.initialize();
      const now = new Date();
      const timestamp = now.toLocaleString('de-DE', { dateStyle: 'short', timeStyle: 'short' });

      if (!this.client) {
        throw new Error('Custom OpenAI client not initialized - missing API key');
      }

      // Calculate total prompt tokens including musthavePrompt
      const totalPromptTokens = await calculateTotalPromptTokens(
        prompt + musthavePrompt // Combined system prompt
      );

      // Calculate available tokens
      const maxTokens = Number(config.tokenLimit);
      const reservedTokens = totalPromptTokens + Number(config.responseTokens);
      const availableTokens = maxTokens - reservedTokens;

      // Truncate content if necessary
      const truncatedContent = await truncateToTokenLimit(content, availableTokens);

      // Make API request
      const response = await this.client.chat.completions.create({
        model: config.custom.model,
        messages: [
          {
            role: "system",
            content: prompt + musthavePrompt
          },
          {
            role: "user",
            content: truncatedContent
          }
        ],
        response_format: { type: 'json_object' },
        temperature: 0.3,
      });

      // Handle response
      if (!response?.choices?.[0]?.message?.content) {
        throw new Error('Invalid API response structure');
      }

      // Log token usage
      console.log(`[DEBUG] [${timestamp}] Custom OpenAI request sent`);
      if (response.usage?.total_tokens !== undefined) {
        console.log(`[DEBUG] [${timestamp}] Total tokens: ${response.usage.total_tokens}`);
      } else {
        console.warn('[WARNING] Custom provider did not return usage metrics');
      }

      const usage = response.usage;
      const mappedUsage = usage ? {
        promptTokens: usage.prompt_tokens,
        completionTokens: usage.completion_tokens,
        totalTokens: usage.total_tokens
      } : null;

      let jsonContent = response.choices[0].message.content;
      console.log('[DEBUG] Raw provider content (first 500 chars):', String(jsonContent).slice(0, 500));

      let parsedResponse;
      try {
        const parseResult = cleanAndParseJson(jsonContent);
        parsedResponse = parseResult.parsed;

        if (parseResult.repaired) {
          console.warn('[WARNING] Playground response required auto-repair');
        }
      } catch (error) {
        console.error('Failed to parse JSON response:', error, '\nRaw content:', jsonContent);
        throw new Error('Invalid JSON response from API');
      }

      // Validate response structure
      if (!parsedResponse || !Array.isArray(parsedResponse.tags) || typeof parsedResponse.correspondent !== 'string') {
        throw new Error('Invalid response structure: missing tags array or correspondent string');
      }

      return {
        document: parsedResponse,
        metrics: mappedUsage,
        truncated: truncatedContent.length < content.length
      };
    } catch (error) {
      console.error('Failed to analyze document:', error);
      return {
        document: { tags: [], correspondent: null },
        metrics: null,
        error: error.message
      };
    }
  }

  /**
   * Generate text based on a prompt
   * @param {string} prompt - The prompt to generate text from
   * @returns {Promise<string>} - The generated text
   */
  async generateText(prompt) {
    try {
      this.initialize();

      if (!this.client) {
        throw new Error('Custom OpenAI client not initialized - missing API key');
      }

      const model = config.custom.model;

      const response = await this.client.chat.completions.create({
        model: model,
        messages: [
          {
            role: "user",
            content: prompt
          }
        ],
        temperature: 0.7,
        max_tokens: 128000
      });

      if (!response?.choices?.[0]?.message?.content) {
        throw new Error('Invalid API response structure');
      }

      return response.choices[0].message.content;
    } catch (error) {
      console.error('Error generating text with Custom OpenAI:', error);
      throw error;
    }
  }

  async checkStatus() {
    try {
      this.initialize();

      if (!this.client) {
        throw new Error('Custom OpenAI client not initialized - missing API key');
      }

      const model = config.custom.model;

      const response = await this.client.chat.completions.create({
        model: model,
        messages: [
          {
            role: "user",
            content: 'Ping'
          }
        ],
        temperature: 0.7,
        max_tokens: 1000
      });

      if (!response?.choices?.[0]?.message?.content) {
        return { status: 'error' };
      }

      return { status: 'ok', model: model };
    } catch (error) {
      console.error('Error generating text with Custom OpenAI:', error);
      return { status: 'error' };
    }
  }
}

module.exports = new CustomOpenAIService();
