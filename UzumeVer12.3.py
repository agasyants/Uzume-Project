import tensorflow as tf
import pretty_midi as pm
import midi_utils as mu
import numpy as np
import glob, os

files = glob.glob('../Normal/*.mid')

def print_pitch(pitch):
    a = []
    for i in range(len(pitch)):
        if pitch[i]==1:
            a.append(i)
    print('pitch:', a)


seq_len = 20

offset_len = 32
count_len = 12
duration_len = 32
disc = 8
dist_len = 2*disc

all_pitches = []
all_pitch_vectors = []
all_note_vectors = []
all_counts = []
all_count_vectors = []
all_offsets = []
all_offset_vectors = []
all_duration_vectors = []
all_durations = []
all_chords = []
all_dist = []
all_densities = []
all_frequencies = []
for file in files[:300]:
    try:
        midi_file = pm.PrettyMIDI(file)
        print(file)
    except:
        continue
    durations = [[0]*duration_len]*seq_len
    duration_vectors = [0]*seq_len
    pitches = [[0]*128]*seq_len
    note_vectors = [[0]*12]*seq_len
    pitch_vectors = [[0]*11]*seq_len
    offset_vectors = [0]*seq_len
    offsets = [[0]*offset_len]*seq_len
    counts = [0]*seq_len
    count_vectors = [0]*seq_len
    dists = [[0]]*seq_len
    bar_vectors= [[0]*12]*seq_len
    frequencies = [0]*seq_len
    densities = [0]*seq_len
    notes = mu.get_unique_notes(mu.round_notes(mu.get_notes_from_track(midi_file),disc))
    bar_vectors.extend(mu.get_bars_vectors(notes))
    print("len(bar_vectors):", len(bar_vectors))
    densities.extend(mu.get_densities(notes))
    frequencies.extend(mu.get_frequencies(notes))
    pitch = [0]*128
    duration = 0
    for x, note in enumerate(notes[:-1]):
        offset = round((notes[x+1].start - note.start)*disc)
        if offset == 0:
            pitch[note.pitch] = 1
            duration += note.end-note.start
        else:
            pitch[note.pitch] = 1
            offset = mu.limit(offset,offset_len)
            offset_vectors.append(offset/offset_len)
            offsets.append(offset-1)
            dists.append([note.start%2/2])
            count = pitch.count(1)
            counts.append(mu.limit(count-1,count_len))
            count_vectors.append(mu.limit(count, count_len)/count_len)
            duration += note.end-note.start
            duration = duration*disc/count
            durations.append(mu.limit(round(duration)-1,duration_len))
            duration_vectors.append(duration/duration_len)
            duration = 0
            pitches.append(pitch)
            note_vectors.append(mu.get_note_vector(pitch))
            pitch_vectors.append(mu.get_pitch_vector(pitch))
            pitch = [0]*128
    print(sum(frequencies)/len(frequencies), sum(densities)/len(densities))
    all_chords.append(bar_vectors)
    all_dist.append(dists)
    all_counts.append(counts)
    all_pitches.append(pitches)
    all_offsets.append(offsets)
    all_densities.append(densities)
    all_durations.append(durations)
    all_frequencies.append(frequencies)
    all_note_vectors.append(note_vectors)
    all_pitch_vectors.append(pitch_vectors)
    all_count_vectors.append(count_vectors)
    all_offset_vectors.append(offset_vectors)
    all_duration_vectors.append(duration_vectors)


def generate_input_x(all):
    input_x = []
    for track in all:
        for i in range(len(track)-seq_len):
            input_x.append(track[i:i+seq_len])
    return np.array(input_x)

def generate_input_y(all):
    input_y = []
    for track in all:
        for i in range(len(track)-seq_len):
            input_y.append(track[i+seq_len])
    return np.array(input_y)


pitch_input_y = generate_input_y(all_pitches)
offset_input_y = generate_input_y(all_offsets)
count_input_y = generate_input_y(all_counts)
count_vector_input_x = generate_input_x(all_count_vectors)
chord_input_x = generate_input_x(all_chords)
chord_input_y = generate_input_y(all_chords)
dist_input_x = generate_input_x(all_dist)
dist_input_y = generate_input_y(all_dist)
duration_input_y = generate_input_y(all_durations)
note_vector_input_x = generate_input_x(all_note_vectors)
pitch_vector_input_x = generate_input_x(all_pitch_vectors)
duration_vector_input_x = generate_input_x(all_duration_vectors)
offset_vector_input_x = generate_input_x(all_offset_vectors)
density_vector_input_y = generate_input_y(all_densities)
frequency_vector_input_y = generate_input_y(all_frequencies)


offset_input = tf.keras.Input((seq_len, 1), name="offset_input")
duration_input =  tf.keras.Input((seq_len, 1), name="duration_input")

dist_input = tf.keras.layers.Input((seq_len, 1), name="dist_input")

note_vector_input =  tf.keras.Input((seq_len, 12), name="note_vector_input")
pitch_vector_input =  tf.keras.Input((seq_len, 11), name="pitch_vector_input")

count_input =  tf.keras.Input((seq_len, 1), name="count_input")

chord_input =  tf.keras.Input((seq_len, 12), name="chord_input")

frequency_add_input = tf.keras.Input((1), name="frequency_add_input")
density_add_input = tf.keras.Input((1), name="density_add_input")
dist_add_input = tf.keras.Input((1), name="dist_add_input")
chord_input_add = tf.keras.Input((12), name="chord_input_add")

input_add = tf.keras.layers.Concatenate()([chord_input_add, dist_add_input, density_add_input, frequency_add_input])


inputs = tf.keras.layers.Concatenate()([offset_input, pitch_vector_input, count_input, 
                                        chord_input, duration_input, note_vector_input, dist_input])
a = tf.keras.layers.LSTM(300, return_sequences=True)(inputs)
b = tf.keras.layers.LSTM(300)(a)
c = tf.keras.layers.Concatenate()([input_add, b])
x = tf.keras.layers.Dense(128)(c)


outputs = {
  'offset_output': tf.keras.layers.Dense(offset_len, name='offset_output', activation="tanh")(x),
  'pitch_output': tf.keras.layers.Dense(128, name='pitch_output', activation="tanh")(x),
  'count_output': tf.keras.layers.Dense(count_len, name='count_output', activation="tanh")(x),
  'duration_output': tf.keras.layers.Dense(duration_len, name='duration_output')(x)
}

model = tf.keras.Model([offset_input, pitch_vector_input, count_input, chord_input, chord_input_add, 
                        dist_input, dist_add_input, duration_input, note_vector_input, 
                        frequency_add_input, density_add_input], outputs)

loss = {
      'pitch_output': tf.keras.losses.CategoricalCrossentropy(from_logits=True),
      'count_output': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      'offset_output': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      'duration_output': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
}

optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

model.compile(loss=loss, optimizer=optimizer)

model.summary()

result = model.fit({"offset_input": offset_vector_input_x,
                    "note_vector_input": note_vector_input_x,
                    "pitch_vector_input": pitch_vector_input_x,
                    "chord_input": chord_input_x,
                    "chord_input_add": chord_input_y,
                    "dist_add_input": dist_input_y,
                    "dist_input": dist_input_x,
                    "duration_input": duration_vector_input_x,
                    "count_input": count_vector_input_x,
                    "frequency_add_input": frequency_vector_input_y,
                    "density_add_input": density_vector_input_y},

                   {"pitch_output": pitch_input_y,
                    "offset_output": offset_input_y,
                    "count_output": count_input_y,
                    "duration_output": duration_input_y}, 

                   batch_size=256, epochs=30, shuffle=True)

model.save('./UzumeSeqVer12.3')


gen = pm.PrettyMIDI()
inst = pm.Instrument(program=0)

start_time = 0
pitch_vector_pred = [[0]*11]*seq_len
note_vector_pred = [[0]*12]*seq_len
count_pred = [0]*seq_len
chord_pred = [[0]*12]*seq_len
dist_pred = [[0]]*(seq_len+1)
offset_vector_pred = [[0]]*seq_len
offset_pred = [[0]*offset_len]*seq_len
duration_vector_pred = [[0]]*seq_len
duration_pred = [[0]*duration_len]*seq_len
chords = []
for i in range(10):
    chords.append(mu.create_chord('Am'))
    chords.append(mu.create_chord('G'))
    chords.append(mu.create_chord('F'))
    chords.append(mu.create_chord('E'))
for i in range(8):
    chords.append(mu.create_chord('Am'))
for i in range(10):
    chords.append(mu.create_chord('F'))
    chords.append(mu.create_chord('G'))
    chords.append(mu.create_chord('Am'))
    chords.append(mu.create_chord('Am'))

density_pred=[]
frequency_pred=[]
for i in range(8):
    density_pred.append(1.0)
    frequency_pred.append(0.24)
for i in range(16):
    density_pred.append(2.5)
    frequency_pred.append(0.5)
for i in range(16):
    density_pred.append(3)
    frequency_pred.append(0.18)
for i in range(40):
    density_pred.append(2.0)
    frequency_pred.append(0.45)
for i in range(8):
    density_pred.append(1.0)
    frequency_pred.append(0.8)
for i in range(160):
    density_pred.append(2.0)
    frequency_pred.append(0.5)


chord_pred.append(chords[0])
j=0
for i in range(2000):
    print(i)
    pred = model.predict({"offset_input": np.expand_dims(offset_vector_pred[i:i+seq_len],0),
                            "note_vector_input": np.expand_dims(note_vector_pred[i:i+seq_len],0),
                            "pitch_vector_input": np.expand_dims(pitch_vector_pred[i:i+seq_len],0),
                            "count_input": np.expand_dims(count_pred[i:i+seq_len],0),
                            "chord_input": np.expand_dims(chord_pred[i:i+seq_len],0),
                            "chord_input_add": np.expand_dims(chord_pred[i+seq_len],0),
                            "dist_input": np.expand_dims(dist_pred[i:i+seq_len],0),
                            "dist_add_input": np.expand_dims(dist_pred[i+seq_len],0),
                            "duration_input": np.expand_dims(duration_vector_pred[i:i+seq_len],0),
                            "density_add_input": np.expand_dims(density_pred[j],0),
                            "frequency_add_input": np.expand_dims(frequency_pred[j],0)})
    pitch = pred["pitch_output"][0]
    add = np.array([0]*128)
    offset = mu.sample(pred["offset_output"][0],t=0.05)+1
    print("offset:",offset)
    offset_vector_pred = np.append(offset_vector_pred, np.expand_dims([offset/offset_len],0), axis=0)
    offset /= disc
    duration = mu.sample(pred["duration_output"][0],t=0.08)+1
    print("duration:",duration)
    duration_vector_pred = np.append(duration_vector_pred, np.expand_dims([duration/duration_len],0), axis=0)
    duration /= disc
    count=mu.sample(pred["count_output"][0],t=0.05)+1
    for i in range(count):
        p=mu.sample(pitch,t=0.02)
        pitch[p]=-1
        add[p]=1
        note = pm.Note(velocity=90, pitch=p, 
                    start=start_time, end=start_time+duration)
        inst.notes.append(note)
    count_pred = np.append(count_pred, count/count_len)
    pitch_vector_pred = np.append(pitch_vector_pred, np.expand_dims(mu.get_pitch_vector(add),0), axis=0)
    note_vector_pred = np.append(note_vector_pred, np.expand_dims(mu.get_note_vector(add),0), axis=0)
    print_pitch(add)
    r1=start_time//2
    start_time += offset
    r2=start_time//2
    dist=start_time%2/2
    dist_pred = np.append(dist_pred, np.expand_dims([dist],0), axis=0)
    if r1!=r2:
        j+=1
        if j>len(chords)-1:
            break
    chord_pred = np.append(chord_pred, np.expand_dims(chords[j],0), axis=0)



gen.instruments.append(inst)
gen.write('./Uver12.3.mid')
print('Uver12.3.mid')