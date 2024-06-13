"""
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

def get_tripwire_name_ID_mapping(tripwire_config, tripwire_names):
  """
  Gets the tripwire IDs from the tripwire names and returns name to ID and ID to name maps

  Keyword arguments:
  endpoint -- the API endpoint of eMDX
  sensorId -- the ID of the camera sensor
  tripwire_names -- the list of names of the tripwire
  """
  tripwires = tripwire_config['tripwires']
  id_to_name, name_to_id  = {}, {}
  for tripwire in tripwires:
    tripwireId = tripwire['id']
    tripwireName = tripwire['name']
    if tripwireName in tripwire_names:
      id_to_name[tripwireId] = tripwireName
      name_to_id[tripwireName] = tripwireId

  return id_to_name, name_to_id

def get_vst_snapshot(snapshot_content):
  """
  Returns opencv image object of the camera sensor snapshot

  Keyword arguments:
  snapshot_content -- the snapshot content retrieved from the API endpoint of VST
  """
  img_array = np.frombuffer(snapshot_content, np.uint8)
  cv2_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
  cv2.destroyAllWindows()
  return cv2_image

def plot_pie_chart(tripwire_results, tripwire_id_map, label):
  pie_distribution = []
  labels = []
  for tripwire_id in tripwire_results:
    crossings = 0
    for direction in tripwire_results[tripwire_id]:
      crossings+=tripwire_results[tripwire_id][direction]
    pie_distribution.append(crossings)
    labels.append(f"{tripwire_id_map[tripwire_id]}: {crossings}")
  plt.pie(pie_distribution, labels = labels)
  plt.legend(title = label)
  plt.show()

def get_all_points(trajs):
    """
    This method accumulates the points in all the trajectories
    """
    xs = []
    ys = []

    for traj in trajs:
        pxs = traj['x'].to_list()
        pys = traj['y'].to_list()

        xs += pxs
        ys += pys
    
    return xs, ys

def compute_heatmap(xs, ys, img_width, img_height, scale):
    """
    This method computes the heatmap of points
    The resolution of generated heatmap can be changed using 'scale' (2 to 10)
    """
    bins = [int(img_width/scale), int(img_height/scale)]
    range=[[0, img_width], [0, img_height]]

    H, _, _ = np.histogram2d(xs, ys, bins=bins, range=range)

    return H

def plot_heatmap(trajs, image, name='heatmap', scale=4, smoothness=2):
    """
    This method plots the heatmap of trajectories
    
    Input Arguments:
        trajs - list of trajectories
        image - sample frame from the video stream
        name - The title of generated plot is taken from 'name'
        scale - controls the resolution of output heatmap (2 to 10)
        smoothness - controls the variance of smoothening (gaussian) filter
    """
    xs, ys = get_all_points(trajs)
    
    img_width, img_height = image.shape[1], image.shape[0]
    H = compute_heatmap(xs, ys, img_width, img_height, scale)
    Hg = gaussian_filter(H, smoothness, truncate=2)

    _, ax = plt.subplots()
    
    ax.imshow(image[::scale, ::scale], alpha=0.99)

    ax.invert_yaxis()
    plt.axis('off')
    ax.imshow(np.ma.masked_where(Hg <= 1e-5, Hg).T, cmap='jet', alpha=0.5)
    ax.set_title(name)
    plt.savefig(f"{name}.png", bbox_inches='tight', pad_inches=0, dpi=280)
    
    return Hg

def plot_stacked_bar_chart(times, object_counts, fromTimestamp, toTimestamp, colors=['b', 'g', 'r', 'c', 'm', 'y', 'k'], xticks_rotation=90):
    
    """
    This method plots the stacked bar chart of object count
    
    Input Arguments:
        times - timerange for which object times to be counted
        object_counts - per object type object counts for each window
        colors - color code for each object type
        xticks_rotation - the angle representing the orientation of labels on x-axis
    """
    
    non_zero_counts_exist = any(any(count!=0 for count in counts) for counts in object_counts.values())
    if non_zero_counts_exist:
        fig, ax = plt.subplots()
        plt.rcParams['figure.figsize'] = [7, 5]
        bottom_offset = [0]* len(times)
        for i, (obj, counts) in enumerate(object_counts.items()):
            ax.bar(times, counts, bottom=bottom_offset, label=obj.capitalize(), color=colors[i%len(colors)])
            bottom_offset = [sum(x) for x in zip(bottom_offset, counts)]

        plt.xlabel('Time')
        plt.ylabel('Counts')
        plt.title(f'Object counts b/w {fromTimestamp} - {toTimestamp}')
        plt.xticks(rotation=xticks_rotation)
        plt.legend()

        plt.show()
