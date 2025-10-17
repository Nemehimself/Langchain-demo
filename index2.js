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

import "dotenv/config";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const loader = new PDFLoader("./data/Cleaning-CV.pdf"); // path to your PDF
const rawCV = await loader.load();

// console.log(rawCV.slice(0, 2));

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 64,
    separator: " "
});

const splitDocs = await splitter.splitDocuments(rawCV);
console.log(splitDocs.slice(0, 2));