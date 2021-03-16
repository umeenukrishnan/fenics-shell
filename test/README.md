To run tests, you need the py.test module.

Just run

  cd <fenics-shells>/test
  py.test-3

Or to run a single test file

  py.test-3 test_reissner_mindlin.py

To run these tests from within the source tree without
needing to install the UFL Python module, update your
PYTHONPATH and PATH by running

  source sourceme.sh

in a bash shell or equivalent. If on other OSes, you
must set the paths whichever way your OS requires.

