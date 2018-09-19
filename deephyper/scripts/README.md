Automatic job script generation and submission
================================================
Edit "runjob.conf" to have the correct environment name and paths

Invoke "python runjob.py" with the arguments:

    python runjob.py <platform> <search-method> <benchmark> <-q queue> <-n nodes> <-t wallminutes>

There are additional args necessary for hyperband or to disable data staging
runjob.py --help will show the possible args

Scripts are generated in the runs/ subdirectory (created here)


On Cooley:
    python runjob.py cooley rs gcn.gcn -q nox11 -n 8 -t 120

On Theta with Postgres DB:
    python runjob.py theta_postgres rs gcn.gcn -q debug-cache-quad -n 5 -t 40

On Theta with SQLite DB:
    python runjob.py theta rs gcn.gcn -q debug-cache-quad -n 5 -t 40
