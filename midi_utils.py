import os
import pretty_midi as pm
import numpy as np


# get all midi files from folder midi-dataset including subfolders
def get_midi_files(path):
    midi_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.mid'):
                midi_files.append(os.path.join(root, file))
    return midi_files

# get all notes from midi file and sor it by start time
def get_notes_from_track(midi_data):
    notes = []
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                notes.append(note)
    notes.sort(key=lambda x: x.start)                
    return notes

def get_notes_from_instrument(instrument):
    notes = []
    for note in instrument.notes:
        notes.append(note)
    notes.sort(key=lambda x: x.start)                
    return notes

# delite same notes
def get_unique_notes(notes):
    uniq = []
    for note in notes:
        is_uniq = True
        for un in uniq:
            if un.pitch == note.pitch and un.start == note.start:
                is_uniq = False
                break
        if is_uniq:
            uniq.append(note)
    return uniq

def uniquness_ratio(notes):
    uniq = []
    for note in notes:
        is_uniq = True
        for un in uniq:
            if un.start == note.start:
                is_uniq = False
                break
        if is_uniq:
            uniq.append(note)
    return len(notes)/len(uniq)

def round_notes(notes,d):
    for note in notes:
        note.start = round(note.start * d) / d
        note.end = round(note.end * d) / d
    return notes

def get_mean_frequency(notes, bpm):
    durations = []
    for i, note in enumerate(notes[:-1]):
        durations.append(notes[i+1].start-note.start)
    mean_duration = np.mean(durations)
    return bpm/mean_duration

# make notes' vector
def get_vector(notes):
    vector = [0] * 12
    for note in notes:
        vector[note.pitch%12] += 1
    vector = np.array(vector, dtype=np.float32)
    if vector.max() == 0:
        return vector
    vector = vector-vector.min()
    vector = vector/vector.mean()/12
    return vector

def get_gain_vector(notes):
    vec = get_vector(notes)
    # calculate each sigmoid
    for i in range(12):
        vec[i] = 1/(1+np.exp(-0.6*(vec[i]*16-8)))
    return vec


# divide notes to bars
def get_bars(notes, bpm):
    spb_bar = 60/bpm*4
    bars = [[]*2 for _ in range(int(notes[len(notes)-1].start//spb_bar))]
    for note in notes:
        note.start = round(note.start * 16) / 16
        note.end = round(note.end * 16) / 16
        bar = int(note.start//spb_bar)
        if bar >= len(bars):
            break
        bars[bar].append(note)
    return bars

# divide notes to bars and normalize starts, ends
def get_bars2(notes, bpm):
    spb_bar = 60/bpm*4
    bars = [[]*2 for _ in range(int(notes[len(notes)-1].start//spb_bar))]
    for note in notes:
        note.start = round(note.start * 16) / 16
        note.end = round(note.end * 16) / 16
        bar = int(note.start//spb_bar)
        if bar >= len(bars):
            break
        duration = note.end-note.start
        note.start %= spb_bar
        note.end = note.start+duration
        bars[bar].append(note)
    return bars

def get_bars3(notes):
    spb_bar = 2
    bars = [[]*2 for _ in range(int(notes[len(notes)-1].start//spb_bar+1))]
    for note in notes:
        bar = int(note.start//spb_bar)
        bars[bar].append(note)
    return bars

def get_bars_vectors(notes):
    bars = get_bars3(notes)
    bars_vectors = []
    for bar in bars:
        s=[]
        for note in bar:
            s.append(note.start)
        bars_vectors.extend([get_gain_vector(bar) for _ in range(len(set(s)))])
    return bars_vectors[:-1]

def get_densities(notes):
    bars = get_bars3(notes)
    densities = []
    for bar in bars:
        s=[]
        for note in bar:
            s.append(note.start)
        densities.extend([(len(s)/len(set(s))) for _ in range(len(set(s)))])
    return densities[:-1]

def get_frequencies(notes):
    bars = get_bars3(notes)
    frequencies = []
    for bar in bars:
        s=[]
        for note in bar:
            s.append(note.start)
        frequencies.extend([2/len(set(s)) for _ in range(len(set(s)))])
    return frequencies[:-1]

def limit(x, limit):
    if x >= limit:
        #print("out of the limit:",x)
        return limit-1
    elif x < 0:
        #print("less that zero:",x)
        return 0
    return int(x)

def get_note_vector(pitch):
    note_vector = [0]*12
    c=sum(pitch)
    for i in range(len(pitch)):
        if pitch[i]==1:
            note_vector[i%12]+=(1/c)
    return note_vector

def get_pitch_vector(pitch):
    note_vector = [0]*11
    c=sum(pitch)
    for i in range(len(pitch)):
        if pitch[i]==1:
            note_vector[i//12]+=(1/c)
    return note_vector


# translating string chords to vector
def create_chord(chord,k=0.2):
    chords = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    vector=[0.05,0,0.05,0,0.05,0.05,0,0.05,0,0.05,0,0.05]
    if chord[-1]=='7':
        chord=chord[:-1]
        a = chords.index(chord[0])+12
        if chords[(a-1)%12][-1]!='#':
            vector[(a-1)%12]=k
        elif chords[(a-2)%12][-1]!='#':
            vector[(a-2)%12]=k
    if chord[-1]=='m':
        n = chords.index(chord[:-1])
        vector[(n+3)%12]=k
    else:
        n = chords.index(chord)
        vector[(n+4)%12]=k
    vector[n]=k+0.1
    vector[(n+7)%12]=k
    return vector

def sample(preds, t = 0.1):
    out = preds
    if np.sum(preds)!=0:
        try:
            preds = preds - np.min(preds)
            preds = preds / np.sum(preds)
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / t
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)
        except:
            return np.argmax(out)
    else:
        return np.argmax(out)
    
class MyNote:
    def __init__(self, note, part):
        self.note = note
        self.part = part

def make_MyNotes(file, disc):
    MyNotes = []
    for instrument in file.instruments[:3]:
        if not instrument.is_drum and instrument.name[-1] != '1':
            notes = get_unique_notes(round_notes(get_notes_from_instrument(instrument), disc))
            for note in notes:
                MyNotes.append(MyNote(note, file.instruments.index(instrument)))
    MyNotes.sort(key=lambda x: x.note.start)
    return MyNotes

def make_notes_from_MyNotes(MyNotes):
    notes = []
    for MyNote in MyNotes:
        notes.append(MyNote.note)
    return notes

def create_OHV(a,b):
    OHV = [0]*b
    OHV[a] = 1
    return OHV