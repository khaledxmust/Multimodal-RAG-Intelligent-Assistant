"""Offline tests of real retrieval/prompt code with external model/service seams replaced."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_backend():
    # These checks intentionally avoid model downloads, GPU allocation and a live database.
    modules = {}
    for name in ('requests', 'qdrant_client', 'retrieval.embeddings', 'retrieval.vector_store', 'config.settings'):
        modules[name] = types.ModuleType(name)
    modules['qdrant_client'].QdrantClient = object
    modules['retrieval.embeddings'].EmbeddingModel = object
    vector_store = modules['retrieval.vector_store']
    vector_store.COLLECTION_NAME = 'test_collection'
    vector_store.get_qdrant_client = lambda: None
    vector_store.get_point_count = lambda client: 0
    settings = modules['config.settings']
    settings.OLLAMA_URL = 'http://localhost:11434'
    settings.OLLAMA_MODEL = 'test_model'
    spec = importlib.util.spec_from_file_location('rag_test_backend', ROOT / 'generation' / 'rag_assistant.py')
    backend = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(backend)
    return backend


class FakeVector(list):
    def tolist(self):
        return list(self)


class FakeEmbedding:
    def encode_queries(self, texts):
        self.texts = texts
        return [FakeVector([1.0, 0.0])]


class FakeClient:
    def __init__(self, points):
        self.points = points

    def query_points(self, **kwargs):
        self.last_query = kwargs
        return types.SimpleNamespace(points=self.points)


class CoreBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = load_backend()

    def test_named_product_reranks_a_relevant_candidate(self):
        general = types.SimpleNamespace(score=0.90, payload={'title': 'General service', 'text': 'Generic information'})
        exact = types.SimpleNamespace(score=0.80, payload={'title': 'WE LIFE', 'text': 'Product details'})
        client = FakeClient([general, exact])
        results, normalized = self.backend.retrieve(client, FakeEmbedding(), 'What is WE LIFE?', limit=1)
        self.assertEqual(results, [exact])
        self.assertEqual(client.last_query['collection_name'], 'test_collection')
        self.assertIn('WE LIFE', normalized)
        # Reranking changes order, while the stored vector similarity keeps its original value.
        self.assertEqual(exact.score, 0.80)

    def test_document_citation_retains_page_and_filename(self):
        point = types.SimpleNamespace(score=0.87, payload={'source_type':'user_document', 'source_name':'demo.pdf', 'page_number':2, 'text':'A sourced demonstration fact.'})
        context, sources = self.backend.build_context([point])
        self.assertIn('Document: demo.pdf', context)
        self.assertIn('Page: 2', context)
        self.assertEqual(sources[0]['page_number'], 2)
        self.assertEqual(self.backend.extract_citation_numbers('One [2]. Another [1]. Repeated [2].'), [2, 1])

    def test_empty_retrieval_produces_empty_context(self):
        client = FakeClient([])
        results, _ = self.backend.retrieve(client, FakeEmbedding(), 'Unseen query')
        self.assertEqual(results, [])
        self.assertEqual(self.backend.build_context(results), ('', []))

    def test_generation_request_uses_configured_local_service(self):
        response = types.SimpleNamespace(ok=True, raise_for_status=lambda: None, json=lambda: {'message': {'content':'Example answer [1].'}, 'done_reason':'stop'})
        with patch.object(self.backend.requests, 'post', create=True, return_value=response) as post:
            answer = self.backend.generate_answer('What is the sample fact?', '[1] A fictional sample fact.')
        self.assertEqual(answer, 'Example answer [1].')
        self.assertEqual(post.call_args.args[0], 'http://localhost:11434/api/chat')
        self.assertEqual(post.call_args.kwargs['json']['model'], 'test_model')
        self.assertFalse(post.call_args.kwargs['json']['think'])

    def test_synthetic_sample_has_unique_ids_and_no_live_sources(self):
        chunks = json.loads((ROOT / 'samples' / 'synthetic_chunks.json').read_text(encoding='utf-8'))
        self.assertEqual(len(chunks), 4)
        self.assertEqual(len({item['chunk_id'] for item in chunks}), len(chunks))
        self.assertEqual({item['content_language'] for item in chunks}, {'ar', 'en'})
        self.assertTrue(all(item['source'] == 'synthetic_demo' and not item['url'] for item in chunks))


if __name__ == '__main__':
    unittest.main()
