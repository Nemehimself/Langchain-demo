//index.js
import dotenv from "dotenv";
dotenv.config();

console.log("API Key loaded:", process.env.OPENROUTER_API_KEY ? "✅ Yes" : "❌ No");

import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

//create the model (the AI brain)
const model = new ChatOpenAI({
  modelName: "gpt-4o-mini", // you can use any OpenRouter-supported model
  temperature: 0.7,
  openAIApiKey: process.env.OPENROUTER_API_KEY,  // Changed from apiKey
  configuration: {
    baseURL: process.env.OPENROUTER_BASE_URL,
    defaultHeaders: {
      "HTTP-Referer": "http://localhost:3000",
      "X-Title": "LangChain Demo",
    },
  },
});

// create a reusable prompt template
const prompt = ChatPromptTemplate.fromTemplate(
    "what are the cities in {country}?"
);

// create an output parser (to get clean text back)
const outputParser = new StringOutputParser();

// chain them together: prompt -> model -> output parser
const chain = prompt.pipe(model).pipe(outputParser);

// run the chain with an example
const result = await chain.invoke({
    country: "Finland",
});

console.log("The cities in Finland are: \n");
console.log(result);