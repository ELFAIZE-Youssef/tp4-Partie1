package ma.emsi.elfaize.test1;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class RagNaif {

    public static void main(String[] args) throws Exception {

        System.out.println("=== Phase 1 : Initialisation et enregistrement des embeddings ===");

        //Instanciation du parser PDF basé sur Apache Tika
        DocumentParser documentParser = new ApacheTikaDocumentParser();

        //Chargement du support de cours RAG en PDF
        Path path = Paths.get("src/main/resources/rag.pdf");
        Document document = FileSystemDocumentLoader.loadDocument(path, documentParser);
        System.out.println(" Fichier chargé : " + path.getFileName());

        //Découpage du document en fragments de texte
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        System.out.println("🔹 Nombre de fragments extraits : " + segments.size());

        //Création du modèle d’embeddings (AllMiniLmL6V2)
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        //Génération des vecteurs d’embedding pour chaque fragment
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        System.out.println("Nombre total d’embeddings créés : " + embeddings.size());

        //Mise en place d’un magasin d’embeddings en mémoire (InMemoryEmbeddingStore)
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        //Association des embeddings et des segments correspondants
        embeddingStore.addAll(embeddings, segments);
        System.out.println("Les embeddings ont été enregistrés avec succès!");

        System.out.println("Phase 1 terminée : préparation du RAG réussie !");
    }
}
