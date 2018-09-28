"""
Basic test for Evaluator : 'local' or 'balsam'.
"""

import json
import logging
from random import randint
import time
import unittest

from deephyper.evaluators import Evaluator

def run(d):
    if 'fail' in d:
        raise RuntimeError
    sleep = d.get('sleep', 0)
    time.sleep(sleep)
    return  d['x1']**2 + d['x2']**2

def key(d):
    x1, x2 = d['x1'], d['x2']
    return json.dumps(dict(x1=x1, x2=x2))

class TestLocal(unittest.TestCase):

    def setUp(self):
        self.ev = Evaluator.create(run, cache_key=key, method='local')

    def tearDown(self):
        for f in self.ev.pending_evals.values(): f.cancel()

    def test_add(self):
        ev = self.ev
        ev.add_eval( dict(ID="test1", x1=3, x2=4) )
        ev.add_eval( dict(ID="test2", x1=3, x2=4) )
        ev.add_eval( dict(ID="test3", x1=10, x2=10) )

        self.assertEqual(len(ev.pending_evals), 2)
        self.assertEqual(len(ev.finished_evals), 0)
        self.assertEqual(len(ev.requested_evals), 3)
        for f in self.ev.pending_evals.values(): f.cancel()

    def test_get_finished_success(self):
        ev = self.ev
        ev.add_eval( dict(ID="test1", x1=3, x2=4) )
        ev.add_eval( dict(ID="test2", x1=3, x2=4) )
        ev.add_eval( dict(ID="test3", x1=10, x2=10) )

        res = []
        while len(res) < 3:
            res.extend(ev.get_finished_evals())

        self.assertEqual(len(res), 3)
        self.assertIn(({'ID':'test1','x1':3,'x2':4}, 25),    res)
        self.assertIn(({'ID':'test2','x1':3,'x2':4}, 25),    res)
        self.assertIn(({'ID':'test3','x1':10,'x2':10}, 200), res)
        for f in self.ev.pending_evals.values(): f.cancel()


if __name__ == "__main__":
    unittest.main()
