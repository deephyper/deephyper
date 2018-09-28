"""
Basic test for Evaluator : 'local' or 'balsam'.
"""
import unittest
from deephyper.evaluators import Evaluator
from deephyper.evaluators.test_functions import run, key

class TestLocal(unittest.TestCase):
    def setUp(self):
        self.ev = Evaluator.create(run, cache_key=key, method='local')

    def tearDown(self):
        for f in self.ev.pending_evals.values(): f.cancel()

    def test_add(self):
        '''ev.add_eval(x) works correctly; caching equivalent evals'''
        ev = self.ev
        # Add 3 evals
        ev.add_eval( dict(ID="test1", x1=3, x2=4) )
        ev.add_eval( dict(ID="test2", x1=3, x2=4) )
        ev.add_eval( dict(ID="test3", x1=10, x2=10) )

        # There are only 2 pending evals (test1==test2)
        self.assertEqual(len(ev.pending_evals), 2)
        self.assertEqual(len(ev.finished_evals), 0)
        self.assertEqual(len(ev.requested_evals), 3)

    def test_add_batch(self):
        '''ev.add_eval_batch(x_list) also works correctly'''
        ev = self.ev
        evals = [
            dict(ID="test1", x1=3, x2=4),
            dict(ID="test2", x1=3, x2=4),
            dict(ID="test3", x1=10, x2=10),
        ]
        ev.add_eval_batch(evals)
        # There are only 2 pending evals (test1==test2)
        self.assertEqual(len(ev.pending_evals), 2)
        self.assertEqual(len(ev.finished_evals), 0)
        self.assertEqual(len(ev.requested_evals), 3)

    def test_get_finished_success(self):
        '''get_finished returns all requested evals'''
        ev = self.ev
        ev.add_eval( dict(ID="test1", x1=3, x2=4) )
        ev.add_eval( dict(ID="test2", x1=3, x2=4) )
        ev.add_eval( dict(ID="test3", x1=10, x2=10) )

        # fetch results until all 3 are finished
        res = []
        while len(res) < 3:
            res.extend(ev.get_finished_evals())

        self.assertEqual(len(ev.finished_evals), 2) # internally, only 2 results
        self.assertEqual(len(res), 3)
        self.assertIn(({'ID':'test1','x1':3,'x2':4}, 25),    res)
        self.assertIn(({'ID':'test2','x1':3,'x2':4}, 25),    res)
        self.assertIn(({'ID':'test3','x1':10,'x2':10}, 200), res)

    def test_get_finished_one_slow(self):
        '''One straggler eval does not block return of all fast ones'''
        ev = self.ev
        # test4 is very slow eval
        ev.add_eval( dict(ID="test4", x1=10, x2=10, sleep=10) )
        ev.add_eval( dict(ID="test1", x1=3, x2=4) )
        ev.add_eval( dict(ID="test2", x1=3, x2=4) )
        ev.add_eval( dict(ID="test3", x1=10, x2=10, sleep=0.1) )

        # fetch results until first 3 are obtained
        res = []
        while len(res) < 3:
            res.extend(ev.get_finished_evals())

        # got all 3 except for the slow one:
        self.assertEqual(len(res), 3)
        self.assertIn(({'ID':'test1','x1':3,'x2':4}, 25),    res)
        self.assertIn(({'ID':'test2','x1':3,'x2':4}, 25),    res)
        self.assertIn(({'ID':'test3','x1':10,'x2':10,'sleep':0.1}, 200), res)

    def test_get_finished_timeout(self):
        '''No exceptions raised if get_finished_evals returns nothing'''
        ev = self.ev
        # two slow evals:
        ev.add_eval( dict(ID="test1", x1=3, x2=4, sleep=10) )
        ev.add_eval( dict(ID="test2", x1=3, x2=4, sleep=10) )

        # non-blocking; no problem:
        res = list(ev.get_finished_evals())
        self.assertEqual(len(res), 0)

    def test_get_finished_fail(self):
        '''If one eval fails, then the result is marked as Evaluator.FAIL_RETURN_VALUE'''
        ev = self.ev
        ev.add_eval( dict(ID="test1", x1=3, x2=4, fail=True) )
        ev.add_eval( dict(ID="test2", x1=3, x2=4, fail=False) )

        res = []
        while len(res) < 2:
            res.extend(ev.get_finished_evals())

        # Got both results, but one is float_max:
        self.assertEqual(len(res), 2)
        yvals = sorted([r[1] for r in res])
        self.assertListEqual(yvals, [25, Evaluator.FAIL_RETURN_VALUE])

    def test_await_success(self):
        '''await_evals(list) works correctly'''
        ev = self.ev
        evals = [
            dict(ID="test1", x1=3, x2=4),
            dict(ID="test2", x1=3, x2=4),
            dict(ID="test3", x1=10, x2=10),
        ]
        ev.add_eval_batch(evals)
        res = list(ev.await_evals(evals))

        self.assertEqual(len(ev.finished_evals), 2) # internally, only 2 results
        self.assertEqual(len(res), 3)
        self.assertIn(({'ID':'test1','x1':3,'x2':4}, 25),    res)
        self.assertIn(({'ID':'test2','x1':3,'x2':4}, 25),    res)
        self.assertIn(({'ID':'test3','x1':10,'x2':10}, 200), res)

    def test_await_one_failed(self):
        '''await_evals returns expected value for failed evals'''
        ev = self.ev
        evals = [
            dict(ID="test1", x1=3, x2=4),
            dict(ID="test2", x1=3, x2=4, fail=True),
            dict(ID="test3", x1=10, x2=10),
        ]
        ev.add_eval_batch(evals)
        res = list(ev.await_evals(evals))

        self.assertEqual(len(ev.finished_evals), 3)
        self.assertEqual(len(res), 3)
        self.assertIn(({'ID':'test1','x1':3,'x2':4}, 25),    res)
        self.assertIn(({'ID':'test2','x1':3,'x2':4,'fail':True}, Evaluator.FAIL_RETURN_VALUE), res)
        self.assertIn(({'ID':'test3','x1':10,'x2':10}, 200), res)

    def test_await_empty(self):
        '''await_evals accepts empty list; does not block'''
        ev = self.ev
        evals = []
        ev.add_eval_batch(evals)
        res = list(ev.await_evals(evals))
        self.assertEqual(len(res), 0)

    def test_await_timeout(self):
        '''TimeoutError is raised on await with timeout'''
        ev = self.ev
        evals = [
            dict(ID="test1", x1=3, x2=4, sleep=0.1),
            dict(ID="test2", x1=3, x2=4, sleep=20),
            dict(ID="test3", x1=10, x2=10),
        ]
        ev.add_eval_batch(evals)
        with self.assertRaises(TimeoutError):
            res = list(ev.await_evals(evals,timeout=3))

        res = list(ev.get_finished_evals())
        self.assertEqual(len(ev.finished_evals), 2)
        self.assertEqual(len(res), 2)
        self.assertIn(({'ID':'test1','x1':3,'x2':4,'sleep':0.1}, 25),    res)
        self.assertNotIn(({'ID':'test2','x1':3,'x2':4,'sleep':20}, 25),    res)
        self.assertIn(({'ID':'test3','x1':10,'x2':10}, 200), res)

if __name__ == "__main__":
    unittest.main()
