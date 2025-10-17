// Ask a question about your PDF and get an AI-generated answer from it
import "dotenv/config";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { ChatOpenAI } from "@langchain/openai";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";

// Load PDF document
const loader = new PDFLoader("./data/Cleaning-CV.pdf");

// load and extract text from PDF
const rawCV = await loader.load();
console.log(`✅ Loaded ${rawCV.length} pages from the PDF.\n`);

// Split the document into smaller chunks
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 64,
    separator: " "
});

const splitDocs = await splitter.splitDocuments(rawCV);

// ✅ Use local embeddings (no API key needed!)
console.log("🔄 Creating embeddings (this may take a moment on first run)...\n");
const embeddings = new HuggingFaceTransformersEmbeddings({
    model: "Xenova/all-MiniLM-L6-v2", // Fast, lightweight model
});

// Convert chunks into embeddings
const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
);

// ✅ Create retriever
const retriever = vectorStore.asRetriever();

// ✅ Create Chat Model using OpenRouter
const model = new ChatOpenAI({
  modelName: "openai/gpt-4o-mini",
  temperature: 0.3,
  openAIApiKey: process.env.OPENROUTER_API_KEY,
  configuration: {
    baseURL: "https://openrouter.ai/api/v1",
  },
});

// ✅ Define the prompt for your QA system
const prompt = ChatPromptTemplate.fromTemplate(`
You are an assistant that answers questions based only on the provided PDF context.

Context:
{context}

Question:
{question}
`);

// Combine retriever + model
const chain = await createStuffDocumentsChain({
  llm: model,
  prompt,
});

//Ask a question about your pdf
const question = "what is the phone number mentioned in the CV?";
console.log(`🔍 Retrieving relevant context for: "${question}"\n`);
const retrievedDocs = await retriever.invoke(question);

console.log("🤖 Generating answer...\n");
const answer = await chain.invoke({
  context: retrievedDocs,
  question,
});

console.log(`Question: ${question}`);
console.log(`Answer: ${answer}`);