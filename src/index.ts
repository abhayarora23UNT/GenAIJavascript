import readline from "readline";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatMessageHistory } from "@langchain/community/stores/message/in_memory";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import "dotenv/config";

// Define constants.
const EMBEDDING_MODEL = "text-embedding-3-large";
const GPT_MODEL = "gpt-4-turbo-preview";
const HISTORY_KEY = "history";
const USER_QUERY_KEY = "input";

// Initialize Pinecone client and index.
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY as string,
});
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME as string);

// This function returns a model used to convert text into a vector (a mathematical representation)//
export const getEmbeddingModel = (): OpenAIEmbeddings => {
  return new OpenAIEmbeddings({
    model: EMBEDDING_MODEL,
    apiKey: process.env.OPEN_AI_KEY,
  });
};

// Function to return a ChatOpenAI model.
export const getChatOpenAI = (): ChatOpenAI => {
  return new ChatOpenAI({
    apiKey: process.env.OPEN_AI_KEY,
    model: GPT_MODEL,
    maxTokens: 2000,
    temperature: 0.7,
  });
};

// Function to add multiple context documents to Pinecone
export const storeContextInPinecone = async () => {
  const embeddings = getEmbeddingModel();
  const vectorStore = await PineconeStore.fromExistingIndex(embeddings, { pineconeIndex });

  // Sample context documents representing different parts of a person's life
  const documents = [
    {
      id: "medical-history",
      pageContent: "Patient has a history of hypertension and diabetes.",
      metadata: { category: "medical", updatedAt: new Date().toISOString() },
    },
    {
      id: "dietary-preferences",
      pageContent: "User prefers vegetarian food, avoids dairy products.",
      metadata: { category: "dietary", updatedAt: new Date().toISOString() },
    },
    {
      id: "exercise-routine",
      pageContent: "User does cardio three times a week and practices yoga.",
      metadata: { category: "exercise", updatedAt: new Date().toISOString() },
    },
    {
      id: "personal-goals",
      pageContent: "User's goal is to lose 10 pounds in 3 months.",
      metadata: { category: "personal", updatedAt: new Date().toISOString() },
    },
  ];

  // Add each document to Pinecone
  await vectorStore.addDocuments(documents);
};

// Function to retrieve context from Pinecone.
// This function retrieves the most relevant context based on a query (like a user’s question).
export const retrieveContextFromPinecone = async (query: string): Promise<string> => {
  const embeddings = getEmbeddingModel();
  const vectorStore = await PineconeStore.fromExistingIndex(embeddings, { pineconeIndex });

  // Perform a similarity search on Pinecone to find the most relevant documents for the query
  const results = await vectorStore.similaritySearch(query, 5); // Fetch top 5 most relevant documents

  // Combine the content of the most relevant documents into a single string
  const relevantContext = results.map((document) => document.pageContent).join("\n");

  return relevantContext;
};

// Function to create a ChatPromptTemplate.
// Use the `ChatPromptTemplate.fromMessages` method to include placeholders for `context` and `history`.
export const getPromptTemplate = (): ChatPromptTemplate => {
  return ChatPromptTemplate.fromMessages([
    ["system", "You are a dedicated medical assistant, focused on providing personalized advice related to health, nutrition, and exercise. Each response should be tailored to the provided context: {context}. Offer clear, actionable recommendations without any unnecessary explanations"],
    new MessagesPlaceholder("history"),
    ["user", "{input}"],
  ]);
};

// Function to create a RunnableWithMessageHistory.
// This function sets up the chat chain, which is a sequence of tasks: first, it prepares the prompt, then it generates a response from the AI, and finally, it parses the output.
export const getChainWithHistory = () => {
  const messageHistory = new ChatMessageHistory();
  const chatModel = getChatOpenAI();
  const prompt = getPromptTemplate();
  return new RunnableWithMessageHistory({
    runnable: prompt.pipe(chatModel).pipe(new StringOutputParser()),
    inputMessagesKey: USER_QUERY_KEY,
    historyMessagesKey: HISTORY_KEY,
    getMessageHistory: (_sessionId) => messageHistory,
  });
};

// Implement the main chat function.
// Use `readline` for user input and the chat chain for generating AI responses.
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function askQuestion(question: string) {
  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      resolve(answer);
    });
  });
}

export const startDocAssistantChat = async (): Promise<void> => {
  const config = { configurable: { sessionId: "1" } }; // Configuration for the chain
  const userInput = await askQuestion("You: ") as string; // Get user input
  const context = await retrieveContextFromPinecone(userInput); // Load context from Pinecone
  const chainWithHistory = getChainWithHistory();      // Get the chat chain with history

  // Invoke the chain with the user input and context
  const output = await chainWithHistory.invoke(
    { input: userInput, context: context },
    config
  );
  console.log("Assistant:", output);

  // Recursive call to continue the chat
  await startDocAssistantChat();
};

// If this script is the main module, start the chat.
// Do not modify this part of the code.
if (require.main === module) {
  startDocAssistantChat().catch((error) => {
    console.error("An error occurred:", error);
  });
}
