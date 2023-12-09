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


model = tf.keras.models.load_model('./UzumeSeqVer12.3')


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
for i in range(4):
    chords.append(mu.create_chord('Am'))
    chords.append(mu.create_chord('Em'))
    chords.append(mu.create_chord('G'))
    chords.append(mu.create_chord('D'))
    chords.append(mu.create_chord('Am'))
    chords.append(mu.create_chord('Em'))
    chords.append(mu.create_chord('G'))
    chords.append(mu.create_chord('B'))
    


density_pred=[]
frequency_pred=[]
for i in range(8):
    density_pred.append(1.8+0.02*i)
    frequency_pred.append(0.6-0.02*i)
for i in range(100):
    density_pred.append(3.2)
    frequency_pred.append(0.42)



chord_pred.append(chords[0])
j=0
for i in range(2000):
    print(i)
    print(chord_pred[-1])
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
    offset = mu.sample(pred["offset_output"][0],t=0.03)+1
    print("offset:",pred["offset_output"][0])
    offset_vector_pred = np.append(offset_vector_pred, np.expand_dims([offset/offset_len],0), axis=0)
    offset /= disc
    duration = np.argmax(pred["duration_output"][0])+1
    print("duration:",duration)
    duration_vector_pred = np.append(duration_vector_pred, np.expand_dims([duration/duration_len],0), axis=0)
    duration /= disc
    count=mu.sample(pred["count_output"][0],t=0.01)+1
    for i in range(count):
        p=mu.sample(pitch,t=0.005)
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