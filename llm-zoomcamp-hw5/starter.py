"""Starter code for the monitoring homework.

Sets up the text-search RAG from homework 1 and a shared OpenAI client.
"""

from openai import OpenAI
import pandas as pd

from gitsource import GithubRepositoryDataReader
from minsearch import Index

from rag_helper import RAGBase
from dotenv import load_dotenv

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

provider = TracerProvider()

trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-zoomcamp")

# exporter = InMemorySpanExporter()
# provider.add_span_processor(
#     SimpleSpanProcessor(exporter)
# )

exporter = InMemorySpanExporter()
provider.add_span_processor(
    SimpleSpanProcessor(exporter)
)
provider.add_span_processor(
    SimpleSpanProcessor(ConsoleSpanExporter())
)

import sqlite3
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class SQLiteSpanExporter(SpanExporter):

    def __init__(self, db_path="traces.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                name TEXT,
                start_time INTEGER,
                end_time INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL
            )
        """)
        self.conn.commit()

    def export(self, spans):
        for span in spans:
            attrs = dict(span.attributes or {})
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,
                    span.start_time,
                    span.end_time,
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                ),
            )
        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.conn.close()

    def force_flush(self):
        return True
    
provider.add_span_processor(
    SimpleSpanProcessor(SQLiteSpanExporter("traces.db"))
)

COMMIT = "8c1834d"

# --- Load the course lessons (same as HW1, HW2, HW4) ---
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id=COMMIT,
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]

index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(documents)

class RAGTraced(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracer = tracer

    def search(self, query, num_results=5):
        with tracer.start_as_current_span("search") as span:
            search_result = self.index.search(query, num_results=num_results)
            #span.set_attribute("query", query)
        return search_result

    def llm(self, prompt):
        with tracer.start_as_current_span("llm") as span:
            input_messages = [
                {'role': 'developer', 'content': self.instructions},
                {'role': 'user', 'content': prompt}
            ]

            response = self.llm_client.responses.create(
                model=self.model,
                input=input_messages
            )
            span.set_attribute("input_tokens", response.usage.input_tokens)
            span.set_attribute("output_tokens", response.usage.output_tokens)

        return response

    def rag(self, query):
        with tracer.start_as_current_span("rag") as span:
            search_results = self.search(query)
            prompt = self.build_prompt(query, search_results)
            response = self.llm(prompt)
            #span.set_attribute("query", query)
        return response.output_text


if __name__ == "__main__":
    load_dotenv()
    client = OpenAI()
    rag = RAGTraced(index=index, llm_client=client)
        
    query = "How does the agentic loop keep calling the model until it stops?"
    answer = rag.rag(query)
    print(answer)

    
    for span in exporter.get_finished_spans():
        duration_ns = span.end_time - span.start_time
        duration_ms = duration_ns / 1_000_000

        print(f"{span.name}: {duration_ms:.2f} ms")

    conn = sqlite3.connect("traces.db")
    cursor = conn.cursor()

    df = pd.read_sql_query("SELECT * FROM spans", conn)

    print(df)
    conn.close()
