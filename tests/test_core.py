"""Offline contract tests: no real images, model downloads, or VLM inference."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from recog.cli import create_parser, import_pairs, main
from recog.graph import build_graph
from recog.matching import run_league
from recog.retrieval import retrieve_top3
from recog.runtime import MiniCPMBackend, parse_json_answer
from recog.schemas import Concept, ContrastGraph, PairEvidence
from recog.storage import load_database, load_support, save_database, write_json


def evidence(a, b, prompt=None):
    return PairEvidence(a.name, b.name, (f'{a.name} cue',), (f'{b.name} cue',))


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.concepts = [Concept('a', 'a.img', (1., 0.)), Concept('b', 'b.img', (.8, .6)),
                         Concept('c', 'c.img', (0., 1.)), Concept('d', 'd.img', (-1., 0.))]

    def test_top3_and_cyclic_league_tie(self):
        candidates = retrieve_top3((10., 0.), self.concepts)
        self.assertEqual([c.concept.name for c in candidates], ['a', 'b', 'c'])
        graph = build_graph(self.concepts, evidence)
        calls = []
        def judge(query, a, b, prompt):
            calls.append((a.name, b.name))
            return {('a', 'b'): 'a', ('a', 'c'): 'c', ('b', 'c'): 'b'}[(a.name, b.name)]
        result = run_league('query.img', candidates, graph, judge)
        self.assertEqual(result.wins, {'a': 1, 'b': 1, 'c': 1})
        self.assertEqual(result.winner, 'a')
        self.assertEqual(len(calls), 3)
        self.assertEqual(graph.evidence('b', 'a').a_to_b, ('b cue',))

    def test_missing_edge_stops_before_model_calls(self):
        candidates = retrieve_top3((1., 0.), self.concepts)
        graph = ContrastGraph({c.name: c for c in self.concepts})
        calls = []
        with self.assertRaises(KeyError):
            run_league('query.img', candidates, graph, lambda *a: calls.append(a))
        self.assertEqual(calls, [])

    def test_retrieval_validation_and_shortlist(self):
        for query in ((0., 0.), (float('nan'), 1.), (1.,)):
            with self.assertRaises(ValueError):
                retrieve_top3(query, self.concepts)
        candidate = retrieve_top3((1., 0.), self.concepts[:1])
        graph = build_graph(self.concepts[:1], evidence)
        self.assertEqual(run_league('q', candidate, graph, None).winner, 'a')

    def test_json_parser_and_r2p_selection(self):
        self.assertEqual(parse_json_answer('```json\n{"Answer":"B"}\n```')['Answer'], 'B')
        with self.assertRaises(ValueError):
            parse_json_answer('unstructured answer')
        from r2p.prompt_generator import BasePromptGenerator
        backend = MiniCPMBackend.__new__(MiniCPMBackend)
        backend.reasoner = SimpleNamespace(prompt_generator=BasePromptGenerator())
        prompts = []
        def chat(images, prompt):
            prompts.append((images, prompt))
            return {'Answer': 'B'}
        backend.chat = chat
        candidates = retrieve_top3((1., 0.), self.concepts)
        winner, _ = backend.select_r2p('query', candidates, {c.name: {'general': 'cue'} for c in self.concepts})
        self.assertEqual(winner, 'b')
        self.assertIn('description', prompts[0][1])
        backend.chat = lambda *args: {'Answer': 'not an option'}
        with self.assertRaises(ValueError):
            backend.select_r2p('query', candidates, {c.name: {'general': 'cue'} for c in self.concepts})


class StorageTests(unittest.TestCase):
    def test_r2p_import_and_directional_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('a', 'b'):
                (root / f'{name}.img').touch()
            source = root / 'source.json'
            source.write_text(json.dumps({'concept_dict': {
                '<a>': {'image': ['a.img'], 'info': {'general': 'one'}},
                'b': {'info': 'two'},
            }, 'path_to_concept': {'b.img': 'b'}}))
            records = load_support(source, r2p=True)
            self.assertEqual([r.name for r in records], ['a', 'b'])
            self.assertEqual(records[1].info, {'general': 'two'})
            self.assertEqual(records[0].image, str(root / 'a.img'))
            concepts = [Concept(r.name, r.image, (1., 0.)) for r in records]
            (root / 'a_vs_b_distinct.json').write_text(json.dumps({'raw_output': json.dumps({
                'a_distinct_features': ['cue A'], 'b_distinct_features': ['cue B']})}))
            graph = import_pairs(concepts, root)
            self.assertEqual(graph.evidence('b', 'a').a_to_b, ('cue B',))
            db = root / 'db.json'
            save_database(db, graph, {r.name: r.info for r in records}, 'encoder')
            restored, descriptions, encoder = load_database(db)
            self.assertEqual(restored.evidence('a', 'b').b_to_a, ('cue B',))
            self.assertEqual(encoder, 'encoder')
            self.assertEqual(descriptions['a']['general'], 'one')
            with self.assertRaises(FileExistsError):
                save_database(db, graph, {}, 'encoder')
            with contextlib.redirect_stdout(io.StringIO()):
                main(['validate', '--database', str(db)])

    def test_duplicate_ids_and_missing_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'a.img').touch()
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'concepts': [{'name': 'a', 'image': 'a.img'}] * 2}))
            with self.assertRaises(ValueError):
                load_support(manifest)
            manifest.write_text(json.dumps({'concepts': [{'name': 'a', 'image': 'missing.img'}]}))
            with self.assertRaises(FileNotFoundError):
                load_support(manifest)

    def test_cli_wiring_without_model_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'image.img').touch()
            manifest = root / 'manifest.json'
            write_json(manifest, {'concepts': [{'name': n, 'image': 'image.img',
                       'info': {'general': n}} for n in ('a', 'b', 'c')]})
            class FakeEncoder:
                def __init__(self, *args): pass
                def encode(self, paths, batch_size=8):
                    return [(1., 0.) for _ in paths]
            backend = SimpleNamespace(extract=evidence, judge=lambda graph: lambda q, a, b, p: a.name,
                                      select_r2p=lambda q, c, d: ('a', {'Answer': 'A'}))
            database = root / 'db.json'
            with patch('r2p.retriever.ClipEncoder', FakeEncoder), patch('recog.cli.release_encoder'), \
                 patch('recog.cli.make_backend', return_value=backend), contextlib.redirect_stdout(io.StringIO()):
                main(['build', '--manifest', str(manifest), '--output', str(database)])
                for method in ('recog', 'r2p'):
                    output = root / f'{method}.json'
                    main(['recognize', '--database', str(database), '--query', str(root / 'image.img'),
                          '--method', method, '--output', str(output)])
                    self.assertEqual(json.loads(output.read_text())['winner'], 'a')


if __name__ == '__main__':
    unittest.main()
