// PDF CHAT APPLICATION
// This app lets me ask questions about a PDF and get AI-generated answers

import dotenv from "dotenv";
dotenv.config();

console.log("API Key loaded:", process.env.OPENROUTER_API_KEY ? "✅ Yes" : "❌ No");

// Import required modules
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { ChatOpenAI } from "@langchain/openai";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import readline from "readline";
import fs from "fs";
import path from "path";

// 🔧 CONFIGURATION

const DATA_FOLDER = "./data";
const CHUNK_SIZE = 500;
const CHUNK_OVERLAP = 50;
const TOP_K_RESULTS = 3; // Number of chunks to retrieve

// STEP 1A: SET UP EMBEDDINGS MODEL
console.log("🔄 Initializing embedding model...");
const embeddings = new HuggingFaceTransformersEmbeddings({
  model: "Xenova/all-MiniLM-L6-v2",
});

// STEP 1B: SET UP CHAT MODEL
const llm = new ChatOpenAI({
  openAIApiKey: process.env.OPENROUTER_API_KEY,
  configuration: {
    baseURL: "https://openrouter.ai/api/v1",
    defaultHeaders: {
      "HTTP-Referer": "http://localhost:3000",
      "X-Title": "Multi-PDF Chat Demo",
    },
  },
  model: "google/gemini-2.0-flash-exp:free",
  temperature: 0.3,
});

// 🆕 STEP 2: SCAN AND LOAD ALL PDFs
// This is the NEW feature - load multiple PDFs!
console.log(`\n📂 Scanning "${DATA_FOLDER}" for PDFs...`);

// Check if data folder exists
if (!fs.existsSync(DATA_FOLDER)) {
  console.error(`❌ Error: "${DATA_FOLDER}" folder not found!`);
  console.log("💡 Please create a 'data' folder and add your PDF files.");
  process.exit(1);
}

// Find all PDF files
const pdfFiles = fs
  .readdirSync(DATA_FOLDER)
  .filter((file) => file.toLowerCase().endsWith(".pdf"));

if (pdfFiles.length === 0) {
  console.error(`❌ No PDF files found in "${DATA_FOLDER}"!`);
  console.log("💡 Please add at least one PDF file to the data folder.");
  process.exit(1);
}

console.log(`✅ Found ${pdfFiles.length} PDF file(s):`);
pdfFiles.forEach((file, i) => console.log(`   ${i + 1}. ${file}`));

// STEP 3: LOAD ALL PDFs
console.log("\n📄 Loading all PDFs...");
let allDocs = [];

for (const file of pdfFiles) {
  const loader = new PDFLoader(path.join(DATA_FOLDER, file));
  const docs = await loader.load();
  
  // Add metadata to track which PDF each chunk came from
  docs.forEach(doc => {
    doc.metadata.source = file;
  });
  
  allDocs.push(...docs);
  console.log(`   ✓ Loaded: ${file} (${docs.length} page(s))`);
}

console.log(`✅ Total: ${allDocs.length} document page(s) loaded`);

// STEP 4: SPLIT INTO CHUNKS
console.log("\n✂️ Splitting documents into chunks...");
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: CHUNK_SIZE,
  chunkOverlap: CHUNK_OVERLAP,
});

const splitDocs = await splitter.splitDocuments(allDocs);
console.log(`✅ Created ${splitDocs.length} text chunks`);

// STEP 5: CREATE VECTOR STORE
console.log("\n🧠 Creating embeddings and vector store...");
const vectorstore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);

// 🆕 Create a retriever (automatic context fetching)
const retriever = vectorstore.asRetriever({
  k: TOP_K_RESULTS, // Return top 3 most relevant chunks
});

console.log("✅ Vector store ready!");

// 🆕 STEP 6: CREATE RAG CHAIN
// This is a cleaner way to combine retrieval + generation

// Define the prompt template
const promptTemplate = ChatPromptTemplate.fromTemplate(`
You are a helpful assistant analyzing PDF documents. Answer questions based ONLY on the context provided.

Context from PDFs:
{context}

Question: {question}

Instructions:
- Be accurate with dates, numbers, and facts
- If the answer involves calculations, show your work
- If the information isn't in the context, say "I don't see that information in the documents"
- When relevant, mention which document the information came from

Answer:`);

// 🆕 Build the RAG chain
const ragChain = RunnableSequence.from([
  {
    // Step 1: Retrieve relevant chunks
    context: async (input) => {
      const docs = await retriever.invoke(input.question);
      return docs.map((d) => `[From: ${d.metadata.source}]\n${d.pageContent}`).join("\n\n");
    },
    // Step 2: Pass through the question
    question: (input) => input.question,
  },
  // Step 3: Apply the prompt template
  promptTemplate,
  // Step 4: Send to LLM
  llm,
  // Step 5: Parse output as string
  new StringOutputParser(),
]);

// STEP 7: INTERACTIVE CHAT INTERFACE
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

console.log("\n" + "=".repeat(60));
console.log("🎉 Multi-PDF Chat is ready!");
console.log(`📚 Loaded ${pdfFiles.length} PDF(s) with ${splitDocs.length} chunks`);
console.log("=".repeat(60));

const askQuestion = async () => {
  rl.question("\n❓ Ask a question (or press Enter to quit): ", async (query) => {
    if (!query.trim()) {
      console.log("\n👋 Goodbye!");
      rl.close();
      return;
    }

    console.log("🔎 Searching documents...");
    
    try {
      // 🆕 Use the RAG chain instead of manual steps
      const answer = await ragChain.invoke({ question: query });
      
      console.log("\n💬 Answer:");
      console.log(answer);
      console.log("\n" + "─".repeat(60));
      
    } catch (error) {
      console.error("\n❌ Error:", error.message);
      console.log("─".repeat(60));
    }

    askQuestion();
  });
};

askQuestion();