"""Ground Truth Experiment

Evaluates all speech files of all speech corpora on all PDAs in the
DataSet('Ground Truth Experiment'), and saves the resulting pitch
tracks in DataSet('Ground Truth Data').

"""

from algos import algos
from tqdm import tqdm
from runforrest import TaskList, defer
import jbof
import warnings
import numpy
from scipy.interpolate import interp1d

# speech datasets:
FDA = jbof.DataSet('FDA/')
KEELE = jbof.DataSet('KEELE/')
KEELE_mod = jbof.DataSet('KEELE_mod/')
PTDB_TUG = jbof.DataSet('PTDB_TUG/')
MOCHA_TIMIT = jbof.DataSet('MOCHA_TIMIT/')
CMU_Arctic = jbof.DataSet('CMU_Arctic/')
try: # TIMIT might not be available
    TIMIT = jbof.DataSet('TIMIT/')
except:
    TIMIT = None

tasklist = TaskList('ground truth experiment', noschedule_if_exist=True, logfile='ground truth experiment.log')

# calculate pitches for all algorithms and all sound files:

for datasetname, dataset in dict(keele=KEELE,
                                 keele_mod=KEELE_mod,
                                 ptdb_tug=PTDB_TUG,
                                 fda=FDA,
                                 timit=TIMIT,
                                 mocha_timit=MOCHA_TIMIT,
                                 cmu_arctic=CMU_Arctic).items():
    for item in tqdm(dataset.all_items(), desc=f'preparing {datasetname}'):
        signal = defer(item).signal
        for algo in algos:
            task = defer(algo, signal, signal.metadata['samplerate'])
            tasklist.schedule(task, metadata=dict(item=item,
                                                  dataset=datasetname,
                                                  algo=algo))

for task in tqdm(tasklist.run(nprocesses=16, autokill=600), smoothing=0, desc='processing'):
    pass  # works for all items and algos except YIN and TIMIT for SX136GCS0 and SI572DMT0

# collect all pitches into a new dataset:

dataset = jbof.create_dataset('ground truth data')
for task in tqdm(tasklist.done_tasks(), desc='collecting pitches'):
    source_item = task.metadata['item']
    if isinstance(source_item, str):
        print(task._id, type(task._id), source_item, task.metadata['algo'].__name__)
    metadata = source_item.metadata
    metadata['speech_dataset'] = task.metadata['dataset']
    metadata['noise_dataset'] = None
    metadata['algo'] = task.metadata['algo'].__name__
    metadata['speech'] = source_item.name
    itemname = f'{metadata["speech_dataset"]}_{metadata["algo"]}_{metadata["speech"]}'
    if dataset.has_item(itemname):
        dataset.delete_item(itemname)
    item = dataset.add_item(name=itemname,
                            metadata=metadata)
    results = task.returnvalue
    item.add_array('time', results[0].astype('float32'))
    item.add_array('pitch', results[1].astype('float32'))
    item.add_array('probability', results[2].astype('float32'))

# add all ground truths to the same dataset:

for corpusname, corpus in dict(keele=KEELE,
                               keele_mod=KEELE_mod,
                               ptdb_tug=PTDB_TUG,
                               fda=FDA).items():
    # no TIMIT or MOCHA_TIMIT or CMU_Arctic here, since they do not include a ground truth pitch
    for source_item in tqdm(corpus.all_items(), desc='collecting ground truths'):
        metadata = source_item.metadata
        metadata['speech_dataset'] = corpusname
        metadata['noise_dataset'] = None
        metadata['algo'] = 'ground_truth'
        metadata['speech'] = source_item.name
        itemname = f'{metadata["speech_dataset"]}_{metadata["algo"]}_{metadata["speech"]}'
        if dataset.has_item(itemname):
            dataset.delete_item(itemname)
        item = dataset.add_item(name=itemname,
                                metadata=metadata)
        item.add_array('time', source_item.pitch['time'].astype('float32'))
        item.add_array('pitch', source_item.pitch['pitch'].astype('float32'))
        item.add_array('probability', (source_item.pitch['pitch'] > 0).astype('float32'))

# add a consensus pitch track for the same dataset:
for corpusname, corpus in dict(ptdb_tug=PTDB_TUG,
                               keele=KEELE,
                               keele_mod=KEELE_mod,
                               fda=FDA,
                               timit=TIMIT,
                               mocha_timit=MOCHA_TIMIT,
                               cmu_arctic=CMU_Arctic).items():
    for source_item in tqdm(list(corpus.all_items()), desc='collecting consensus pitches'):
        matching_items = dataset.find_items(speech_dataset=corpusname, speech=source_item.name)

        signal = source_item.signal
        seconds = len(signal)/signal.metadata['samplerate']

        reference_time = numpy.linspace(0, seconds, seconds*1000)
        pitches = []
        probs = []

        good_algos = ['bana', 'crepe', 'dio', 'dnn', 'kaldi', 'pefac',
                      'praat', 'rapt', 'sacc', 'safe', 'sift', 'shr', 'srh',
                      'straight', 'swipe', 'mbsc', 'nls2', 'yaapt', 'yin']
        for item in matching_items:
            if item.metadata['algo'] not in good_algos:
                continue
            time, pitch, prob = item.time, item.pitch, item.probability
            pitches.append(interp1d(time, pitch, copy=False, bounds_error=False, fill_value=0)(reference_time))
            probs.append(interp1d(time, prob, copy=False, bounds_error=False, fill_value=0)(reference_time))

        pitches = numpy.array(pitches)
        probs = numpy.array(probs)

        with warnings.catch_warnings():
            # ignore nan-related warnings:
            warnings.filterwarnings("ignore",
                                    message="invalid value encountered in (greater|less)",
                                    category=RuntimeWarning)
            warnings.filterwarnings("ignore",
                                    message="All-NaN slice encountered",
                                    category=RuntimeWarning)
            warnings.filterwarnings("ignore",
                                    message="Mean of empty slice",
                                    category=RuntimeWarning)

            # exclude unvoiced pitches:
            pitches[probs < 0.5] = numpy.nan
            pitches[pitches == 0] = numpy.nan

            consensus_median = numpy.nanmedian(pitches, axis=0)
            consensus_mask = numpy.abs(1 - pitches/consensus_median) < 0.2
            consensus_prob = numpy.sum(consensus_mask, axis=0) / len(good_algos)

            # exclude pitches outside of consensus range:
            pitches[~consensus_mask] = numpy.nan

            # convert to regular numpy arrays:
            consensus_pitch = numpy.nanmean(pitches, axis=0)
            consensus_pitch[numpy.isnan(consensus_pitch)] = 0
            consensus_prob[numpy.isnan(consensus_prob)] = 0

        metadata = source_item.metadata
        metadata['speech_dataset'] = corpusname
        metadata['noise_dataset'] = None
        metadata['algo'] = 'consensus'
        metadata['speech'] = source_item.name
        itemname = f'{metadata["speech_dataset"]}_{metadata["algo"]}_{metadata["speech"]}'
        if dataset.has_item(itemname):
            dataset.delete_item(itemname)
        item = dataset.add_item(name=itemname,
                                metadata=metadata)
        item.add_array('time', reference_time.astype('float32'))
        item.add_array('pitch', consensus_pitch.astype('float32'))
        item.add_array('probability', consensus_prob.astype('float32'))
