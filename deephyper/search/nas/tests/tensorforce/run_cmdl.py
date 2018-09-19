from subprocess import call

interp_path = '/Users/Deathn0t/anaconda2/envs/balsam/bin/python'
loc_script = '/Users/Deathn0t/Documents/Argonne/deephyper/search/nas/tests/tensorforce/'
l = ['polynome_2', 'ackley_', 'dixonprice_', 'levy_', 'griewank_']

for fw in l:
    call([interp_path, f'{loc_script}/ppo_mathfun_2D.py', fw])
