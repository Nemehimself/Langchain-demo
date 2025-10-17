// import "dotenv/config";
// import { GithubRepoLoader } from "langchain/document_loaders/web/github";

// const loader = new GithubRepoLoader(
//   "https://github.com/langchain-ai/langchainjs",
//   {
//     recursive: false,
//     ignorePaths: ["*.md", "yarn.lock"],
//   }
// );

// const docs = await loader.load();
// console.log(docs.slice(0, 3));

// import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";

// const loader = new CheerioWebBaseLoader("https://nairaland.com");
// const docs = await loader.load();

// console.log("Page loaded:", docs[0].metadata.title);
// console.log(docs[0].pageContent.slice(0, 200));

//Load a pdf file, split into chunks and print on the console

import "dotenv/config";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

async function main() {
  try {

// Load PDF document
const loader = new PDFLoader("./data/Cleaning-CV.pdf"); // path to your PDF

// load and extract text from PDF
const rawCV = await loader.load();
console.log(`✅ Loaded ${rawCV.length} pages from the PDF.\n`);

// console.log(rawCV.slice(0, 2));

// Split the document into smaller chunks
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 64,
    separator: " "
});

const splitDocs = await splitter.splitDocuments(rawCV);

// Display the first two chunks
console.log("First two chunks:\n");
splitDocs.slice(0, 2).forEach((chunk, index) => {
    console.log(`chunk ${index + 1}:\n`);
    console.log(chunk.pageContent); // The text from the document
    console.log("\n---\n");
    // Accessing properties:
    // console.log(chunk.metadata);     // Information about the document
    // console.log(chunk.metadata.loc.pageNumber);  // Which page this is from
});

  } catch (error) {
    console.error("Error loading or processing PDF:", error);
  }
}
main();