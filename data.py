import cPickle
import datetime
import os

classes = [
    'c34.0',
    'c34.1',
    'c34.2',
    'c34.3',
    'c34.9',
    'c50.1',
    'c50.2',
    'c50.3',
    'c50.4',
    'c50.5',
    'c50.8',
    'c50.9'
]

print( 'Loading data...' )
print( datetime.datetime.now() )

# load data from pickle
f = open( 'data.pkl', 'r' )

classes = cPickle.load( f )
chars = cPickle.load( f )
char_indices = cPickle.load( f )
indices_char = cPickle.load( f )

maxlen = cPickle.load( f )
step = cPickle.load( f )

X = cPickle.load( f )
y = cPickle.load( f )

f.close()

