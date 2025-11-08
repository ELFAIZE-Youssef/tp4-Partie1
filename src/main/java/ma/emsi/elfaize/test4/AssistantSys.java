package ma.emsi.elfaize.test4;

import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;

public interface AssistantSys {
    @SystemMessage("""
Tu es un assistant intelligent qui peut s'appuyer sur le RAG pour répondre aux questions liées à tes domaines d'expertise (IA, RAG, fine-tuning, etc.).
Si la question de l'utilisateur ne concerne pas ces sujets (par exemple : salutations, discussions générales, météo, etc.),
réponds simplement et naturellement sans utiliser le RAG.
""")

    String chat(@UserMessage String message);

}
