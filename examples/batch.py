"""
Contributors: salvadordura@gmail.com
"""

import datetime
import sys
import json
import logging
from time import sleep, time
from copy import copy
from subprocess import Popen, PIPE
from inspyred import ec   # evolutionary algorithm
from random import Random # pseudorandom number generation
from mpi4py import MPI
from numpy import linspace

def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

# Define MPI message tags
tags = enum('READY', 'DONE', 'HOLD', 'EXIT', 'START')

# required to make json saving work in Python 2/3
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

# -------------------------------------------------------------------------------
# function to create a folder if it does not exist
# -------------------------------------------------------------------------------
def createFolder(folder):
    import os
    if not os.path.exists(folder):
        try:
            os.mkdir(folder)
        except OSError:
            print(' Could not create %s' %(folder))


# -------------------------------------------------------------------------------
# function to convert tuples to strings (avoids erro when saving/loading)
# -------------------------------------------------------------------------------
def tupleToStr (obj):
    #print '\nbefore:', obj
    if type(obj) == list:
        for item in obj:
            if type(item) in [list, dict]:
                tupleToStr(item)
    elif type(obj) == dict:
        for key,val in obj.items():
            if type(val) in [list, dict]:
                tupleToStr(val)
            if type(key) == tuple:
                obj[str(key)] = obj.pop(key) 
    #print 'after:', obj
    return obj

def saveJSON(fileName, data):
    import json, io
    with io.open(fileName, 'w', encoding='utf8') as fileObj:
        str_ = json.dumps(data,
                          indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)
        fileObj.write(to_unicode(str_))

def bashTemplate():
    ''' return the bash commands required by template for batch simulation'''
    
    return """#!/bin/bash 
%s
cd %s
%s
    """
# -------------------------------------------------------------------------------
# Batch class
# -------------------------------------------------------------------------------
class Batch(object):

    def __init__(self, cfgFile='cfg.py', params=None, groupedParams=None, initCfg={}, seed=None):
        self.batchLabel = 'batch_'+str(datetime.date.today())
        self.cfgFile = cfgFile
        self.initCfg = initCfg
        self.saveFolder = '/'+self.batchLabel
        self.method = 'grid'
        self.runCfg = {}
        self.evolCfg = {}
        self.params = []
        self.cfg = params

        self.seed = seed
        if params:
            for k,v in params.items():
                self.params.append({'label': k, 'initial': v['initial'],
                                    'lower_bound': v['minval'],
                                    'upper_bound': v['maxval']})
        if groupedParams:
            for p in self.params:
                if p['label'] in groupedParams: p['group'] = True
    

    def save(self, filename):
        import os
        from copy import deepcopy
        basename = os.path.basename(filename)
        folder = filename.split(basename)[0]
        ext = basename.split('.')[1]
        
        # make dir
        createFolder(folder)

        odict = deepcopy(self.__dict__)
        if 'evolCfg' in odict:
            odict['evolCfg']['fitnessFunc'] = 'removed'
        dataSave = {'batch': tupleToStr(odict)} 
        if ext == 'json':
            #from json import encoder
            #encoder.FLOAT_REPR = lambda o: format(o, '.12g')
            print(('Saving batch to %s ... ' % (filename)))
            saveJSON(filename, dataSave)

    def saveScripts(self):
        import os
        import imp

        # create Folder to save simulation
        createFolder(self.saveFolder)
        
        # save Batch dict as json
        targetFile = self.saveFolder+'/'+self.batchLabel+'_batch.json'
        self.save(targetFile)

        # copy this batch script to folder
        targetFile = self.saveFolder+'/'+self.batchLabel+'_batchScript.py'
        os.system('cp ' + os.path.realpath(__file__) + ' ' + targetFile) 
        
        os.system('cp ' + os.path.realpath(__file__) + ' ' + self.saveFolder + '/batchScript.py')
        
        # save initial seed
        with open(self.saveFolder + '/_seed.seed', 'w') as seed_file:
            if not self.seed: self.seed = int(time())
            seed_file.write(str(self.seed))


    def openFiles2SaveStats(self):
        stat_file_name = '%s/%s_stats.cvs' %(self.saveFolder, self.batchLabel)
        ind_file_name = '%s/%s_stats_indiv.cvs' %(self.saveFolder, self.batchLabel)
        individual = open(ind_file_name, 'w')
        stats = open(stat_file_name, 'w')
        stats.write('#gen  pop-size  worst  best  median  average  std-deviation\n')
        individual.write('#gen  #ind  fitness  [candidate]\n')
        return stats, individual


    def run(self):
        global ngen
        ngen = -1
 
        def evaluator(candidates, args):
            import os
            import signal
            global ngen
            ngen += 1

            # paths to required scripts
            genFolderPath = self.saveFolder + '/gen_' + str(ngen)

            # mpi command setup
            nodes = args.get('nodes', 1)
            paramLabels = args.get('paramLabels', [])

            # create folder if it does not exist
            createFolder(genFolderPath)

            num_workers = nodes

            print("Master starting %d tasks over %d workers" % (len(candidates), num_workers))

            fitness = [None for cand in candidates]
            total_jobs = len(candidates)
            holding_workers = 0
            candidate_index = 0  # index given out to jobs
            jobs_completed = 0  # index received from jobs
            while holding_workers < num_workers:
                data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                source = status.Get_source()
                tag = status.Get_tag()

                if tag == tags.READY:
                    # Worker is ready, so send it a task
                    if candidate_index < total_jobs:
                        params_data = {}
                        for label, value in zip(paramLabels, candidates[candidate_index]):
                            params_data[label] = value
                            #print('set %s=%s' % (label, value))
                        # add the task_index to params for writing to disk
                        params_data['task_index'] = candidate_index
                        comm.send(params_data, dest=source, tag=tags.START)
                        print("Sending task %d to worker %d" % (candidate_index, source))
                        candidate_index += 1
                    else:
                        comm.isend(None, dest=source, tag=tags.HOLD)
                        holding_workers += 1

                elif tag == tags.DONE:
                    #print("Got data from worker %d" % source)
                    finished_candidate = data[1][2]  # task_index
                    fitness[finished_candidate] = data[1][0]  # rmse
                    jobs_completed += 1
                    print('  Candidate %d fitness = %.1f' % (finished_candidate, fitness[finished_candidate]))

            # prepare for the next round by signalling holding workers
            comm.bcast(None, root=0)


            print("-"*80)
            print("  Completed a generation  ")
            print("-"*80)
            return fitness
            

        # -------------------------------------------------------------------------------
        # Evolutionary optimization: Generation of first population candidates
        # -------------------------------------------------------------------------------
        def generator(random, args):
            # generate initial values for candidates
            #return args.get('initial')
            return [random.uniform(l, u) for l, u in zip(args.get('lower_bound'), args.get('upper_bound'))]
        # -------------------------------------------------------------------------------
        # Mutator
        # -------------------------------------------------------------------------------
        @ec.variators.mutator
        def nonuniform_bounds_mutation(random, candidate, args):
            """Return the mutants produced by nonuniform mutation on the candidates.
            .. Arguments:
                random -- the random number generator object
                candidate -- the candidate solution
                args -- a dictionary of keyword arguments
            Required keyword arguments in args:
            Optional keyword arguments in args:
            - *mutation_strength* -- the strength of the mutation, where higher
                values correspond to greater variation (default 1)
            """
            lower_bound = args.get('lower_bound')
            upper_bound = args.get('upper_bound')
            strength = args.setdefault('mutation_strength', 1)
            mutant = copy(candidate)
            for i, (c, lo, hi) in enumerate(zip(candidate, lower_bound, upper_bound)):
                if random.random() <= 0.5:
                    new_value = c + (hi - c) * (1.0 - random.random() ** strength)
                else:
                    new_value = c - (c - lo) * (1.0 - random.random() ** strength)
                mutant[i] = new_value
            
            return mutant

        # -------------------------------------------------------------------------------
        # Evolutionary optimization: Main code
        # -------------------------------------------------------------------------------
        import os

        # Initializations and preliminaries
        comm = MPI.COMM_WORLD   # get MPI communicator object
        status = MPI.Status()   # get MPI status object

        # create main sim directory and save scripts
        self.saveScripts()
        
        # log for simulation      
        logger = logging.getLogger('inspyred.ec')
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(self.saveFolder+'/inspyred.log', mode='a')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)    

        # create randomizer instance
        rand = Random()
        rand.seed(self.seed) 
        
        # create file handlers for observers
        stats_file, ind_stats_file = self.openFiles2SaveStats()

        # gather **kwargs
        kwargs = {'cfg': self.cfg}
        kwargs['num_inputs'] = len(self.params)
        kwargs['paramLabels'] = [x['label'] for x in self.params]
        kwargs['initial'] = [x['initial'] for x in self.params]
        kwargs['lower_bound'] = [x['lower_bound'] for x in self.params]
        kwargs['upper_bound'] = [x['upper_bound'] for x in self.params]
        kwargs['statistics_file'] = stats_file
        kwargs['individuals_file'] = ind_stats_file
        kwargs['netParamsSavePath'] = self.saveFolder+'/'+self.batchLabel+'_netParams.py'

        for key, value in self.evolCfg.items(): 
            kwargs[key] = value
        if not 'maximize' in kwargs: kwargs['maximize'] = False
        
        for key, value in self.runCfg.items(): 
            kwargs[key] = value

        ####################################################################
        #                       Evolution strategy
        ####################################################################
        # Custom algorithm based on Krichmar's params
        if self.evolCfg['evolAlgorithm'] == 'custom':
            ea = ec.EvolutionaryComputation(rand)
            ea.selector = ec.selectors.tournament_selection
            ea.variator = [ec.variators.uniform_crossover, nonuniform_bounds_mutation] 
            ea.replacer = ec.replacers.generational_replacement
            if not 'tournament_size' in kwargs: kwargs['tournament_size'] = 2
            if not 'num_selected' in kwargs: kwargs['num_selected'] = kwargs['pop_size']
        
        # Genetic
        elif self.evolCfg['evolAlgorithm'] == 'genetic':
            ea = ec.GA(rand)
        
        # Evolution Strategy
        elif self.evolCfg['evolAlgorithm'] == 'evolutionStrategy':
            ea = ec.ES(rand)
        
        # Simulated Annealing
        elif self.evolCfg['evolAlgorithm'] == 'simulatedAnnealing':
            ea = ec.SA(rand)
        
        # Differential Evolution
        elif self.evolCfg['evolAlgorithm'] == 'diffEvolution':
            ea = ec.DEA(rand)
        
        # Estimation of Distribution
        elif self.evolCfg['evolAlgorithm'] == 'estimationDist':
            ea = ec.EDA(rand)
        
        # Particle Swarm optimization
        elif self.evolCfg['evolAlgorithm'] == 'particleSwarm':
            from inspyred import swarm 
            ea = swarm.PSO(rand)
            ea.topology = swarm.topologies.ring_topology
        
        # Ant colony optimization (requires components)
        elif self.evolCfg['evolAlgorithm'] == 'antColony':
            from inspyred import swarm
            if not 'components' in kwargs: raise ValueError("%s requires components" %(self.evolCfg['evolAlgorithm']))
            ea = swarm.ACS(rand, self.evolCfg['components'])
            ea.topology = swarm.topologies.ring_topology
        
        else:
            raise ValueError("%s is not a valid strategy" %(self.evolCfg['evolAlgorithm']))
        ####################################################################
        ea.terminator = ec.terminators.generation_termination
        ea.observer = [ec.observers.stats_observer, ec.observers.file_observer]
        # -------------------------------------------------------------------------------
        # Run algorithm
        # ------------------------------------------------------------------------------- 
        final_pop = ea.evolve(generator=generator, 
                            evaluator=evaluator,
                            bounder=ec.Bounder(kwargs['lower_bound'],kwargs['upper_bound']),
                            logger=logger,
                            **kwargs)

        # close file
        stats_file.close()
        ind_stats_file.close()
    
        # print best and finish
        print(('Best Solution: \n{0}'.format(str(max(final_pop)))))
        print("-"*80)
        print("   Completed evolutionary algorithm parameter optimization   ")
        print("-"*80)

        num_workers = kwargs['nodes']

        for worker in range(1,num_workers):
            comm.send(None, dest=worker, tag=tags.EXIT)

        closed_workers = 0
        while closed_workers < num_workers:
            comm.recv(source=MPI.ANY_SOURCE, tag=tags.EXIT, status=status)
            source = status.Get_source()
            closed_workers += 1
            print("Worker %d exiting (%d running)" % (source, num_workers - closed_workers))
       
        #sys.exit()
