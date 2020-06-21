# -*- coding: utf-8 -*-
import datetime
import json
import networkx as nx
import numpy as np
import os
import osmnx as ox
import pickle
import pandas as pd
import requests

from sklearn.externals import joblib
from sklearn.svm import SVC

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

import azureml.core
from azureml.core import Workspace
from azureml.core.dataset import Dataset
from azureml.core.model import Model


def getDist(start, end):
  """ Returns shortest path given tuples
  start: (lat,lon) tuple
  end: (lat,lon) tuple
  """
  orig = ox.get_nearest_node(G, start)
  dest = ox.get_nearest_node(G, end)
  if nx.has_path(G, orig, dest):
    route = nx.shortest_path(G, orig, dest, weight='travel_time')
    edge_lengths = ox.utils_graph.get_route_edge_attributes(G, route, 'length')
    return sum(edge_lengths)
  else:
    raise ValueError('Path not foun') 


GRAPH_FILE_PATH = "https://grab5033896937.blob.core.windows.net/azureml/Dataset/grab/singapore.graphml"

try:
    # load workspace configuration from the config.json file in the current folder.
    #ws = Workspace.from_config()

    ws = Workspace.get(name="<<Insert Name>>",
                       subscription_id="<<Insert Subscription Id>>",
                       resource_group="<<Insert Resource Group>>")

    dataset = Dataset.get_by_name(ws, 'sg_graphml')

    # list the files referenced by sg_graphml dataset
    GRAPH_FILE_PATH = dataset.to_path()
    
    G = ox.load_graphml(GRAPH_FILE_PATH)
except:
    G = ox.graph_from_place('Singapore', network_type='drive')
    ox.save_graphml(G, filepath=GRAPH_FILE_PATH)

def init():
    global model
    # Get the path where the deployed model can be found.
    model_path = Model.get_model_path('grab-model-reg')
    model = joblib.load(model_path)


input_sample ={"latitude_origin": -6.141255,
              "longitude_origin": 106.692710,
              "latitude_destination": -6.141150,
              "longitude_destination": -6.141150,
              "timestamp": 1590487113,
              "hour_of_day": 9,
              "day_of_week": 1
            }

output_sample = np.array([0]) # This is a integer type sample. Use the data type that reflects the expected result


@input_schema('data', StandardPythonParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))

def run(data):
    try:
        dist = getDist((data['latitude_origin'], data['longitude_origin']),
                (data['latitude_destination'], data['longitude_destination']))
        day_of_year = int(datetime.date.fromtimestamp(data['timestamp']).strftime('%j'))
        model_input = np.array([dist,
                                data['latitude_origin'],
                                data['longitude_origin'],
                                data['latitude_destination'],
                                data['longitude_destination'],
                                data['hour_of_day'],
                                data['day_of_week'],
                                day_of_year])
        result = model.predict(model_input.reshape(1,-1))
        # Return any datatype as long as it is JSON-serializable
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        error = str(e)
        return json.dumps({"error": result})
