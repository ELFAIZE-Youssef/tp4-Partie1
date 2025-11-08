package ma.emsi.elfaize.test4;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.emsi.elfaize.test1.Assistant;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestPasDeRag {

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) throws Exception {
        configureLogger();
        System.out.println("=== Test 4 - Pas de RAG ===");

        // 1. Initialisation du parser et du splitter
        DocumentParser parser = new ApacheTikaDocumentParser();
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // 2. Chargement du document PDF (RAG)
        Path path = Paths.get("src/main/resources/rag.pdf");
        Document document = FileSystemDocumentLoader.loadDocument(path, parser);
        List<TextSegment> segments = splitter.split(document);
        System.out.println("Document chargé : " + path.getFileName() + " (" + segments.size() + " segments)");

        // 3. Génération et stockage des embeddings
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);

        // 4. Création du ContentRetriever
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // 5. Lecture de la clé Gemini
        String GEMINI_API_KEY = System.getenv("GEMINI_KEY");
        if (GEMINI_API_KEY == null) {
            throw new IllegalStateException("Variable d'environnement GEMINI_KEY non définie !");
        }

        // 6. Création du modèle Gemini
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(GEMINI_API_KEY)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // 7. Classe interne pour décider si on utilise le RAG
        class QueryRouterPourEviterRag implements QueryRouter {

            @Override
            public List<ContentRetriever> route(Query query) {
                String question = "Est-ce que la requête '" + query.text()
                        + "' porte sur le 'RAG' (Retrieval Augmented Generation) ou le 'Fine Tuning' ? "
                        + "Réponds seulement par 'oui', 'non', ou 'peut-être'.";
                String reponse = model.chat(question);
                if (reponse.toLowerCase().contains("non")) {
                    return Collections.emptyList();
                } else {
                    return Collections.singletonList(contentRetriever);
                }
            }

        }

        QueryRouter router = new QueryRouterPourEviterRag();

        // 8. Création du RetrievalAugmentor
        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        // 9. Création de l’assistant
//        Assistant assistant = AiServices.builder(Assistant.class)
//                .chatModel(model)
//                .retrievalAugmentor(augmentor)
//                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
//                .build();

        AssistantSys assistant = AiServices.builder(AssistantSys.class)
                .chatModel(model)
                .retrievalAugmentor(augmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
               .build();


        // 10. Interaction utilisateur
        Scanner scanner = new Scanner(System.in);
        System.out.println("\nPosez vos questions (tapez 'exit' pour quitter)");
        System.out.print("-> ");

        while (true) {
            String question = scanner.nextLine();
            if (question.equalsIgnoreCase("exit")) break;
            String reponse = assistant.chat(question);
            System.out.println("\nRéponse : " + reponse);
            System.out.print("\n> ");
        }

        System.out.println("Fin du test 4 - Pas de RAG terminé !");
    }
}
