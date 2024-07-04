import dotenv
import os
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings


dotenv.load_dotenv()

MODEL = os.environ.get("LLM_MODEL")



graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    enhanced_schema=True
)

graph.refresh_schema()

chat_model = ChatOpenAI(model=MODEL, temperature=0)
qa_model = ChatOpenAI(model=MODEL, temperature=0)



queries_template = """
Task:
Genera delle query Cypher per un database Neo4j.

Istruzioni:
Usa soltanto i tipi di relazioni e le proprietà fornite nello schema.
Non usare altri tipi di relazioni o proprietà che non sono nello schema.
Il nome del prodotto deve essere in maiuscolo.
Il nome del prodotto è contenuto nella proprietà "antecedents" del nodo avente etichetta `lv5Node`.
Dovrai cercare le parole chiave esattamente come sono state fornite. Ad esempio, se ti viene chiesto di cercare la categoria delle bevande,
dovrai cercare la parola chiave "BEVANDE" e non "BEVANDA".
Ricorda che:
- `lv5Node` corrisponde al LIVELLO 5;
- `lv4Node` corrisponde al LIVELLO 4;
- `lv3Node` corrisponde al LIVELLO 3;
- `lv2Node` corrisponde al LIVELLO 2.
Un livello è "più alto" di un altro se il suo valore numerico è inferiore. Ad esempio, il LIVELLO 2 è più alto del LIVELLO 3; e viceversa.


Schema:
{schema}

Note:
Non spiegare le query e non scusarti quando rispondi. Non rispondere a nessuna domanda che chieda qualcosa di diverso dalla costruzione di una query Cypher. 
Non includere alcun testo oltre alla query Cypher generata. Assicurati che la direzione della relazione sia corretta nelle tue query. 
Assicurati di dare un alias a entrambe le entità e alle relazioni in modo appropriato. 
Non eseguire query che aggiungano o cancellino dati dal database. 
Se devi dividere dei numeri, assicurati che il denominatore sia diverso da zero. 
Se non sei in grado di generare una query Cypher, di' che non sai rispondere alla domanda. 
Il prompt è in lingua italiana, e dovresti rispondere in italiano. 
Assicurati di usare IS NULL o IS NOT NULL quando analizzi le proprietà mancanti. Non restituire mai le proprietà di embedding nelle tue query. 
Non devi mai includere la dichiarazione "GROUP BY" nella tua query.
Il tuo obiettivo principale è quello di consigliare dei nuovi prodotti ad un cliente, dato in input un prodotto già acquistato:
devi trovare dei nodi con l'etichetta `lv5Node` dove un attributo `antecedents` contiene una specifica stringa, cioè il nome del prodotto. 
Ricorda che un nodo di livello x non può essere connesso direttamente a un nodo di livello y se il valore assoluto di x-y è diverso da 1.
Se il prodotto da cercare ha più di una parola, dovrai cercare separatamente le parole. 
Ad esempio, se ti viene chiesto di cercare "GOCCIOLE PAVESI", devi cercare separatamente "GOCCIOLE" e "PAVESI", in questo modo: "(product IN p.antecedents WHERE product CONTAINS "GOCCIOLE" AND product CONTAINS "PAVESI")".



Esempi: Qui ci sono un paio di esempi di query Cypher già generate per domande particolari: 

# Avendo comprato PARMALAT, consigliami dei prodotti correlati
MATCH (p:lv5Node)
WHERE ANY (product IN p.antecedents WHERE product CONTAINS "PARMALAT")
MATCH (p)-[:`son of`]->(m:lv4Node)-[:`son of`]->(o:lv3Node)<-[:`son of`]-(m2:lv4Node)<-[:`son of`]-(p2:lv5Node)
WITH p2, [product IN p2.antecedents] AS prodotti_lists
WHERE p2 <> p AND SIZE(prodotti_lists) > 0
UNWIND prodotti_lists AS prodotti_consigliati
RETURN DISTINCT prodotti_consigliati

# Partendo da "KINDER BRIOSS", consigliami dei prodotti che appartengono UNICAMENTE alla categoria SALUMI (LIVELLO 2) oppure alla categoria FORMAGGI (LIVELLO 2)
MATCH (p:lv5Node)
WHERE ANY (product IN p.antecedents WHERE product CONTAINS "KINDER" AND product CONTAINS "BRIOSS")
MATCH (p)-[:`son of`]->(m:lv4Node)-[:`son of`]->(o:lv3Node)-[:`son of`]->(s:lv2Node)<-[:`son of`]-(o2:lv3Node)<-[:`son of`]-(m2:lv4Node)<-[:`son of`]-(p2:lv5Node)
WHERE ANY (product IN s.antecedents WHERE product CONTAINS "SALUMI" OR product CONTAINS "FORMAGGI")
MATCH (p2)
WITH p2, [product IN p2.antecedents WHERE product CONTAINS "SALUMI" OR product CONTAINS "FORMAGGI"] AS prodotti_lists
WHERE p2 <> p AND SIZE(prodotti_lists) > 0
UNWIND prodotti_lists AS prodotti_consigliati
RETURN DISTINCT prodotti_consigliati

# Partendo da "KINDER BRIOSS", consigliami dei prodotti che appartengono UNICAMENTE alla categoria MERENDINE (LIVELLO 3)
MATCH (p:lv5Node)
WHERE ANY (product IN p.antecedents WHERE product CONTAINS "KINDER" AND product CONTAINS "BRIOSS")
MATCH (p)-[:`son of`]->(m:lv4Node)-[:`son of`]->(o:lv3Node)<-[:`son of`]-(m2:lv4Node)<-[:`son of`]-(p2:lv5Node)
WHERE ANY (product IN o.antecedents WHERE product CONTAINS "MERENDINE")
MATCH (p2)
WITH p2, [product IN p2.antecedents WHERE product CONTAINS "MERENDINE"] AS prodotti_lists
WHERE p2 <> p AND SIZE(prodotti_lists) > 0
UNWIND prodotti_lists AS prodotti_consigliati
RETURN DISTINCT prodotti_consigliati

# Partendo da "KINDER BRIOSS", consigliami dei prodotti che appartengono UNICAMENTE alla categoria OLIVE (LIVELLO 4)
MATCH (p:lv5Node)
WHERE ANY (product IN p.antecedents WHERE product CONTAINS "KINDER" AND product CONTAINS "BRIOSS")
MATCH (p)-[:`son of`]->(m:lv4Node)<-[:`son of`]-(p2:lv5Node)
WHERE ANY (product IN m.antecedents WHERE product CONTAINS "OLIVE")
MATCH (p2)
WITH p2, [product IN p2.antecedents WHERE product CONTAINS "OLIVE"] AS prodotti_lists
WHERE p2 <> p AND SIZE(prodotti_lists) > 0
UNWIND prodotti_lists AS prodotti_consigliati
RETURN DISTINCT prodotti_consigliati


# Considera il prodotto "KINDER BRIOSS". Devi consigliare dei prodotti che ASSOLUTAMENTE NON appartengono nè alla categoria SALUMI (LIVELLO 2) nè alla categoria FORMAGGI (LIVELLO 2)
MATCH (p:lv5Node)
WHERE ANY (product IN p.antecedents WHERE product CONTAINS "KINDER" AND product CONTAINS "BRIOSS")
MATCH (p)-[:`son of`]->(m:lv4Node)-[:`son of`]->(o:lv3Node)-[:`son of`]->(s:lv2Node)<-[:`son of`]-(o2:lv3Node)<-[:`son of`]-(m2:lv4Node)<-[:`son of`]-(p2:lv5Node)
MATCH (p2)
WITH p2, [product IN p2.antecedents WHERE NOT product CONTAINS "SALUMI" AND NOT product CONTAINS "FORMAGGI"] AS prodotti_lists
WHERE p2 <> p AND SIZE(prodotti_lists) > 0
UNWIND prodotti_lists AS prodotti_consigliati
RETURN DISTINCT prodotti_consigliati


# Considera il prodotto "KINDER BRIOSS". Devi consigliare dei prodotti che ASSOLUTAMENTE NON appartengono alla categoria MERENDINE (LIVELLO 3)
MATCH (p:lv5Node)
WHERE ANY (product IN p.antecedents WHERE product CONTAINS "KINDER" AND product CONTAINS "BRIOSS")
MATCH (p)-[:`son of`]->(m:lv4Node)-[:`son of`]->(o:lv3Node)-[:`son of`]->(s:lv2Node)<-[:`son of`]-(o2:lv3Node)<-[:`son of`]-(m2:lv4Node)<-[:`son of`]-(p2:lv5Node)
MATCH (p2)
WITH p2, [product IN p2.antecedents WHERE NOT product CONTAINS "MERENDINE"] AS prodotti_lists
WHERE p2 <> p AND SIZE(prodotti_lists) > 0
UNWIND prodotti_lists AS prodotti_consigliati
RETURN DISTINCT prodotti_consigliati


# Considera il prodotto "KINDER BRIOSS". Devi consigliare dei prodotti che ASSOLUTAMENTE NON appartengono alla categoria OLIVE (LIVELLO 4)
MATCH (p:lv5Node)
WHERE ANY (product IN p.antecedents WHERE product CONTAINS "KINDER" AND product CONTAINS "BRIOSS")
MATCH (p)-[:`son of`]->(m:lv4Node)-[:`son of`]->(o:lv3Node)-[:`son of`]->(s:lv2Node)<-[:`son of`]-(o2:lv3Node)<-[:`son of`]-(m2:lv4Node)<-[:`son of`]-(p2:lv5Node)
MATCH (p2)
WITH p2, [product IN p2.antecedents WHERE NOT product CONTAINS "OLIVE"] AS prodotti_lists
WHERE p2 <> p AND SIZE(prodotti_lists) > 0
UNWIND prodotti_lists AS prodotti_consigliati
RETURN DISTINCT prodotti_consigliati


La domanda è la seguente:
{question}



""" 


cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=queries_template
)


qa_generation_template = """
Sei un assistente che prende i risultati
da una query Neo4j Cypher e forma una risposta leggibile dagli umani. La sezione
dei risultati della query contiene i risultati di una query Cypher che è stata
generata in base alla domanda in linguaggio naturale dell'utente. Le informazioni fornite
sono autorevoli, non si deve mai dubitare di esse o cercare di utilizzare le
le proprie conoscenze interne per correggerle. Il tuo output deve sembrare una risposta alla domanda.


Risultati della query:
{context}

Domanda:
{question}

Se il risultato della query è vuoto, devi dire di non conoscere la risposta.
Le informazioni vuote si presentano come segue: []

Se l'informazione non è vuota, si deve fornire una risposta usando i
risultati. Se la domanda riguarda una durata temporale, si supponga che i risultati della query
siano divisi per fasce mensili, a meno che non sia specificato diversamente.

Non dire mai di non sapere la risposta se ci sono dati nei risultati della query.
Utilizza sempre i dati presenti nei risultati della query.

Dopo aver eseguito la query, devi assemblare una lista di prodotti. Se ci sono più di 5 prodotti, restituisci una lista randomica di 5 prodotti.
Ovviamente, non devi includere il prodotto di partenza nella lista di prodotti consigliati.
Tra i prodotti scelti randomicamente devono esserci i prodotti che ti sembrano più simili alla categoria richiesta.
Ad esempio, se nella domanda iniziale era stato chiesto di cercare tra i FORMAGGI e i SALUMI, 
dovresti includere nella lista SOLO dei prodotti nel cui nome è contenuta la parola "FORMAGGI" oppure la parola "SALUMI".
Tra i 5 prodotti scelti dovrai includere tutte le categorie richieste nella domanda iniziale.
Ad esempio, se nella domanda iniziale era stato chiesto di cercare tra i FORMAGGI e i SALUMI, includi nella lista 3 prodotti che contengono la parola "FORMAGGI" e 2 prodotti che contengono la parola "SALUMI", o viceversa.
Devi scrivere nella lista soltanto il nome del prodotto, senza alcun altro testo. Ad esempio, se il prodotto è '1001207       GALBANI GALBANONE 5KG         , FORMAGGI, FORMAGGI BANCO TAGLIO, PASTE FILATE STAGIONATE',
devi includere nella lista soltanto 'GALBANI GALBANONE 5KG'.

Se un prodotto è ripetuto più volte nella lista che hai costruito, devi rimuovere le ripetizioni.

"""

qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template
)



market_cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=chat_model,
    qa_llm=qa_model,
    graph=graph,
    verbose=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    validate_cypher=True,
    top_k=100,
)