import { OpenAI } from "langchain/llms/openai";
import { OpenAIEmbeddings } from "langchain/embeddings/openai"; 
import { PromptTemplate, FewShotPromptTemplate } from "langchain/prompts";
import { initializeAgentExecutorWithOptions } from "langchain/agents";
import { SerpAPI } from "langchain/tools";
import { Calculator } from "langchain/tools/calculator";
import { BufferMemory } from "langchain/memory";
import { ConversationChain, LLMChain, AnalyzeDocumentChain, loadSummarizationChain, loadQAChain } from "langchain/chains";
import * as dotenv from 'dotenv';
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { createClient } from '@supabase/supabase-js';
import * as fs from "fs";
import { Document } from "langchain/document";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { SupabaseHybridSearch } from "langchain/retrievers/supabase";


dotenv.config();

const basic_call = async () => {
    const model = new OpenAI({ temperature: 0.1 });
    const res = await model.call("What is the capital city of France?");

    console.log({ res });
};

const templatePromptCall = async () => {
    const template = "What is the capital city of {country}?";
    const prompt = new PromptTemplate({
        inputVariables: ["country"],
        template: template,
    });

    const promptString = await prompt.format({
        country: "France",
    })

    const model = new OpenAI({ temperature: 0.1});
    const res = await model.call(promptString);

    console.log({ res });
};

const fewShotExamples = async () => {
    const examples = [
        { country: "United States", capital: "Washington D.C."},
        { country: "India", capital: "New Delhi"},
    ];

    const exampleFormatterTemplate = "Country: {country}\nCapital: {capital}\n";
    const examplePrompt = new PromptTemplate({
        inputVariables: ["country", "capital"],
        template: exampleFormatterTemplate,
    });

    console.log("Example Prompt", examplePrompt.format(examples[0]));

    const fewShotPrompt = new FewShotPromptTemplate ({
        examples,
        examplePrompt,
        prefix: "What is the capital city of the country below?",
        suffix: "Country: {country}\nCapital:",
        inputVariables: ["country"],
        exampleSeparator:"\n\n",
        templateFormat: "f-string"
    });

    const fewShotPromptString = await fewShotPrompt.format({country: "France"});

    const model = new OpenAI({ temperature: 0.1});
    const res = await model.call(fewShotPromptString);

    console.log({ res });
};

const agentExample = async () => {
    const model = new OpenAI({ temperature: 0 });

    const tools = [new SerpAPI(), new Calculator()];

    const executor = await initializeAgentExecutorWithOptions(
        tools,
        model,
        {
            agentType: "zero-shot-react-description"
        }
    );

    const input = "What are the total number of countries in Africa raised to the power of 3?";

    const result = await executor.call({ input });
    
    console.log(result.output);
};

const conversationExample = async () => {
    const model = new OpenAI();
    
    const memory = new BufferMemory();
    const chain = new ConversationChain({ llm: model, memory: memory });
    const firstResponse = await chain.call({ input: "Hello, I'm John." });
    const secondResponse = await chain.call({ input: "What's my name?" });
}

const embeddingExample = async () => {
    const embedding = new OpenAIEmbeddings();
    const res = await embedding.embedQuery("Hello world");
    
    const documentRes = await embedding.embedDocuments([
        "Hello world",
        "Bye bye",
    ]);
}

const textSplitterExample = async () => {
    const text = `Hi\n.My name is Kartikey Chauhan.\n\nHow are you?\nOkay then keep your secrets a a a a.
    This is a weird message but I am just testing some things.
    Bye!\n\n`;

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 10,
        chunkOverlap: 1,
    });

    const chunks = splitter.createDocuments([text]);
    console.log(chunks);
}


const vectorDBExample = async () => {
    const supabase = createClient(process.env.SUPABASE_PROJECT_URL!, process.env.SUPABASE_API_KEY!);

    const text = fs.readFileSync(
        require.resolve("./sample.txt"),
        "utf-8"
    );

    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000
    });

    const docs = await textSplitter.createDocuments([text]);

    const embeddings = new OpenAIEmbeddings();
    await addDocuments(docs);

    async function addDocuments(documents: Document[], ids?: string[]): Promise<void> {
        const texts = documents.map(({ pageContent }) => pageContent);
        return addVectors(await embeddings.embedDocuments(texts), documents, ids);
    }

    async function addVectors(vectors: number[][], documents: Document[], ids?: string[]): Promise<void> {
        vectors.forEach(async (vector, idx) => {
            const { data, error } = await supabase.from('documents').upsert({
                id: `${idx}`,
                metadata:  {
                    ...documents[idx].metadata,
                },
                content: documents[idx].pageContent,
                embedding: vector,
            });
        });
    }
}

const vectorDBSimilaritySearch = async () => {
    const supabase = createClient(process.env.SUPABASE_PROJECT_URL!, process.env.SUPABASE_API_KEY!);
    
    const vectorStore = await SupabaseVectorStore.fromExistingIndex(new OpenAIEmbeddings(), {
        client: supabase,
        tableName: "Documents",
        queryName: "match_documents"
    });

    const res = await vectorStore.similaritySearch("He came to come to report on the State of the Union on what year?", 1);
    console.log(res);
}

const vectorDBHybridSearch = async () => {
    const supabase = createClient(process.env.SUPABASE_PROJECT_URL!, process.env.SUPABASE_API_KEY!);
    
    const retriever = new SupabaseHybridSearch(new OpenAIEmbeddings(), {
        client: supabase,
        similarityK: 1,
        keywordK: 1,
        tableName: "documents",
        similarityQueryName: "match_documents",
        keywordQueryName: "kw_match_documents",
      });

    const res = await retriever.getRelevantDocuments("He came to come to report on the State of the Union on what year?");
    console.log(res);
};

const rawTextLoadExample = async () => {
     const loader = new TextLoader(
        "./sample.txt"
     );

     const docs = await loader.load();

     console.log({docs});
     return docs;
}

const basicChain = async () => {
    const model = new OpenAI({ temperature: 0.1 });
    const template = "What is the capital city of {country}?";
    const prompt = new PromptTemplate({
        template,
        inputVariables: ['country']
    });
    const chain = new LLMChain({ llm: model, prompt});
    const res = await chain.call({country: "France"});
    console.log({ res });
}

const summarizeDocsChain = async () => {
    const model = new OpenAI({ temperature: 0.1 });
    const text = "A single piece of text as input. Use AnalyzeDocumentChain when you want to summarize a single piece of information such as this string.";
    
    const combineDocsChain = loadSummarizationChain(model);

    const chain = new AnalyzeDocumentChain({combineDocumentsChain: combineDocsChain});
    const res = await chain.call({
        input_document: text
    });

    console.log({ res });
}

const QAChain = async () => {
    const model = new OpenAI();

    const chain = loadQAChain(model);
    
    const docs = [
      new Document({ pageContent: "Rachel went to Harvard" }),
      new Document({ pageContent: "Tom went to Stanford" }),
    ];

    const res = await chain.call({
      input_documents: docs,
      question: "Where did rachel go to college",
    });

    console.log({ res });
  };