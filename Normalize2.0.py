import numpy as np
import midi_utils as mu
import pretty_midi as pm
import glob, os

class MyNote:
    def __init__(self, note, part):
      self.note = note
      self.part = part

def change_track_pitch(notes, current_key, name):
    inst = pm.Instrument(program=0)
    inst.name = name
    for note in notes:
        pitch = note.pitch - current_key
        if pitch < 0:
            pitch += 12
        inst.notes.append(note)
    return inst

def is_playing_between_times(instrument:pm.Instrument, time1:float, time2:float):
    for note in instrument.notes:
        if (note.start >= time1 and note.start <= time2) or (note.start < time1 and note.end > time2):
            return True
    return False

def is_playing_at_bar(instrument:pm.Instrument, bpm:float, bar:float):
    spb_bar = 60/bpm*4
    return is_playing_between_times(instrument, bar*spb_bar, (bar+1)*spb_bar)

def get_play_time(instrument:pm.Instrument, bpm:float, note:int):
    spb_bar = 60/bpm*4
    is_play = pm.Instrument(program=0,name=instrument.name+'1')
    for i in range(int(instrument.notes[-1].end//spb_bar+1)):
        if is_playing_at_bar(instrument, bpm, i):
            is_play.notes.append(pm.Note(pitch=note, start=i*spb_bar, end=(i+1)*spb_bar, velocity=50))
    return is_play

def find_key(notes):
  pitches = np.array([note[0] for note in notes])
  durations = np.array([note[1] for note in notes])
  pitches = pitches % 12
  [[12,45],[0.23, 1.3]]

  pitch_durations = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0}

  for i, pitch in enumerate(pitches):
    pitch_durations[pitch] += durations[i]

  # print(pitch_durations)
  pitch_durations = list(pitch_durations.values())

  major_profile = [6.35, 2.32, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
  minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

  # calculate relarions
  pitch_durations_mean = np.mean(pitch_durations)

  maj_correlations = []
  min_correlations = []
  # major corelation
  for i in range(12):

    major_profile_mean = np.mean(major_profile) 
    minor_profile_mean = np.mean(minor_profile)

    maj_t_sum = 0.0
    min_t_sum = 0.0
    maj_a_sum = 0.0
    min_a_sum = 0.0
    maj_b_sum = 0.0
    min_b_sum = 0.0
    for i in range(12):
      maj_t_sum += (pitch_durations[i]-pitch_durations_mean)*(major_profile[i]-major_profile_mean)
      min_t_sum += (pitch_durations[i]-pitch_durations_mean)*(minor_profile[i]-minor_profile_mean)

      maj_a_sum += (pitch_durations[i]-pitch_durations_mean)**2
      min_a_sum += (pitch_durations[i]-pitch_durations_mean)**2

      maj_b_sum += (major_profile[i]-major_profile_mean)**2
      min_b_sum += (minor_profile[i]-minor_profile_mean)**2

    maj_correlations.append(maj_t_sum/np.sqrt(maj_a_sum*maj_b_sum))
    min_correlations.append(min_t_sum/np.sqrt(min_a_sum*min_b_sum))

    # shift profile
    major_profile = np.roll(major_profile, 1)
    minor_profile = np.roll(minor_profile, 1)

  return [maj_correlations, min_correlations]


def get_midi_files(path):
    midi_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.mid'):
                midi_files.append(os.path.join(root, file))
    return midi_files


files = get_midi_files('../clean_midi/')
save_to = '../NormalTo3/'

for file in files[:1]:
    try:
        midi = pm.PrettyMIDI(file)
    except:
        print("Error: " + str(file))
        continue
    print('a')
    if len(midi.get_tempo_changes()[1])!=1 or len(midi.instruments)<3:
        continue
    
    # get file name
    file_name = os.path.basename(file)

    bpm = midi.get_tempo_changes()[1][0]
    print(file,bpm)
    
    all_notes = []
    for instrument in midi.instruments:
        if not instrument.is_drum and pm.program_to_instrument_class(instrument.program) != 'Sound Effects' and pm.program_to_instrument_class(instrument.program) != 'Percussive':
            notes = []
            pitches = []
            for note in instrument.notes:
                pitches.append(note.pitch)
                note.start = note.start*bpm/60
                note.end = note.end*bpm/60
                notes.append(note)
                all_notes.append(note)
    all_notes.sort(key=lambda note: note.start)
    
    keys = find_key(list(map(lambda note: [note.pitch, note.end-note.start], all_notes)))

    maj_i = np.argmax(keys[0])
    min_i =np.argmax(keys[1])

    if keys[0][maj_i] > keys[1][min_i]:
        # major
        current_key = maj_i
    else:
        # minor
        current_key = min_i+3
    
    out = pm.PrettyMIDI()
    out.instruments.append(change_track_pitch(all_notes, current_key, "new"))
    out.write(save_to+file_name.replace(".mid","_")+str(bpm)+'_'+str(current_key)+'_normal2.mid')