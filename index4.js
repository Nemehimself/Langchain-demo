// PDF CHAT APPLICATION
// This app lets me ask questions about a PDF and get AI-generated answers

import dotenv from "dotenv";
dotenv.config();

console.log("API Key loaded:", process.env.OPENROUTER_API_KEY ? "✅ Yes" : "❌ No");

import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { ChatOpenAI } from "@langchain/openai";         // For generating AI answers
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import readline from "readline";

// STEP 1A: SET UP EMBEDDINGS MODEL
// Purpose: Converts text chunks into numbers (vectors) for similarity search
// This finds which chunks are most relevant to my question
const embeddings = new HuggingFaceTransformersEmbeddings({
  model: "Xenova/all-MiniLM-L6-v2", // ✅ Runs locally, 100% FREE!
});

// STEP 1B: SET UP CHAT MODEL
// Purpose: Reads the relevant chunks and generates human-like answers
// This is what actually answers my questions
const llm = new ChatOpenAI({
  openAIApiKey: process.env.OPENROUTER_API_KEY,
  configuration: {
    baseURL: "https://openrouter.ai/api/v1",
    defaultHeaders: {
      "HTTP-Referer": "http://localhost:3000", // Required by OpenRouter
      "X-Title": "PDF Chat Demo",
    },
  },
  model: "google/gemini-2.0-flash-exp:free", // FREE model
  temperature: 0.3, // Lower = more accurate for factual questions
});

// STEP 2: LOAD THE PDF FILE
console.log("📄 Loading your PDF...");
const loader = new PDFLoader("./data/Cleaning-CV.pdf");
const rawDocs = await loader.load();
console.log(`✅ Loaded ${rawDocs.length} document(s)`);

// STEP 3: SPLIT PDF INTO SMALLER CHUNKS
// Why? Large documents are hard to process, so we break them into pieces
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,      // Each chunk is ~500 characters
  chunkOverlap: 50,    // Overlap ensures we don't lose context between chunks
});

console.log("✂️ Splitting text into smaller chunks...");
const splitDocs = await splitter.splitDocuments(rawDocs);
console.log(`✅ Split into ${splitDocs.length} chunks`);

// STEP 4: CREATE VECTOR DATABASE
// This converts all chunks to vectors and stores them in memory
// Now we can quickly find relevant chunks using similarity search
console.log("🧠 Creating embeddings and storing them...");
const vectorstore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
console.log("✅ Vector store ready!");

// STEP 5: INTERACTIVE CHAT INTERFACE
// This is where I ask questions and get AI-generated answers
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const askQuestion = async () => {
  rl.question("\n❓ Ask a question about your PDF (or press Enter to quit): ", async (query) => {
    // Exit if user presses Enter without typing anything
    if (!query.trim()) {
      console.log("👋 Goodbye!");
      rl.close();
      return;
    }

    // STEP A: FIND RELEVANT CHUNKS
    // Search the vector database for the 3 most similar chunks
    console.log("🔎 Searching for relevant chunks...");
    const results = await vectorstore.similaritySearch(query, 3);

    // 📝 STEP B: COMBINE CHUNKS INTO CONTEXT
    // Take all the relevant chunks and merge them into one text block
    const context = results.map(doc => doc.pageContent).join("\n\n");

    // 🤖 STEP C: GENERATE AI ANSWER
    // Send the context + question to the LLM to get a natural answer
    console.log("🤖 Generating answer...\n");

    const prompt = `You are a helpful assistant analyzing a PDF document. Answer the question based ONLY on the context provided. Be accurate with dates and numbers.

Context:
${context}

Question: ${query}

Instructions:
- If the question involves dates or math, calculate carefully
- Quote exact text from the context when possible
- If the answer isn't in the context, say "I don't see that information in the document"

Answer:`;

    // This is where the ChatOpenAI model (llm) is actually used!
    const response = await llm.invoke(prompt);
    console.log(`💬 Answer:\n${response.content}\n`);
    console.log("─".repeat(60));

    // console.log("📘 Source chunks used:\n");
    // results.forEach((doc, i) => {
    //   console.log(`Chunk ${i + 1}:\n${doc.pageContent.slice(0, 200)}...\n`);
    // });

    // Ask another question
    askQuestion();
  });
};



// Start the chat interface
console.log("\n" + "=".repeat(60));
console.log("🎉 PDF Chat is ready! Ask me anything about your document.");
console.log("=".repeat(60));
askQuestion();