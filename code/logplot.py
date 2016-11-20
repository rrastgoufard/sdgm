 
from __future__ import print_function

import os
import subprocess
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

def striptime(line, datesep="::", side=-1):
  return line.split(datesep)[side]

def columnbreak(line, sep="  "):
  splits = [l.strip() for l in line.split(sep)]
  return [s for s in splits if s]
    
def epochmb(lines):
  def emb(line):
    e, mbMB = line.split(" ")
    mb, MB = mbMB.split("/")
    return map(int, [e, mb, MB])
  epochs = [e + (1.0*mb/MB) 
            for e, mb, MB in map(emb,lines)]
  return epochs

def datatimes(lines):
  data = []
  times = []
  for l in lines:
    t = columnbreak(striptime(l, side=0))[1]
    d = columnbreak(striptime(l))
    try:
      # If we change descenters in the middle
      # of a run, then we have some non-data
      # information in the middle of the log.
      # If we can parse an epoch mb/MB at the
      # beginning of the line, then this line
      # contains numbers!
      epochmb([d[0]])
      data.append(d)
      times.append(t)
    except:
      continue
    
  return data, times
  
def logtonp(infile):
  
  # Read the whole file into memory.
  with open(infile, "r") as fin:
    lines = fin.readlines()

  # Find specific anchor keywords that we can
  # use to separate the portions of the log.
  endintro = "Created shareds for"
  starting = "Starting!"
  
  importants = [
    endintro, 
    starting,
    ]
  
  # Keep track of where each important keyword
  # is located.
  locs = {}
  for i, line in enumerate(lines):
    for imp in importants:
      if imp in line:
        locs[imp] = i
  
  # A section of settings.  Just return it
  # directly.
  settings = [striptime(line).rstrip() for line in 
              lines[1:locs[endintro]]]
  settings = settingsdict(settings)
  
  # A specific line showing the headings of the
  # columns.
  heading = striptime(lines[locs[starting]+1])
  columns = columnbreak(heading)
  
  # All of the data values stored in the lines.
  startdata = locs[starting]+2
  data, times = datatimes(lines[startdata:])
  columns.append("Time")
  
  # Make one list for each column of data.
  data = list(zip(*data))
  data.append(times)
  
  # Collect the columns into an ordereddict
  collected = OrderedDict()
  for head, dat in zip(columns, data):
    npdat = epochmb(dat) if "Epoch" in head else dat
    head = "Epoch" if "Epoch" in head else head
    collected[head] = np.array(list(map(float,npdat)))
  
  return settings, collected

def logplot(infile):
  print(infile)
  settings, data = logtonp(infile)
  
  #for s in settings:
    #print(s)
    
  fig, axes = plt.subplots(1,len(data),
                           figsize=(72,3))
  
  title = infile.split("../logs/")[1]
  title = title.split(".txt")[0]
  title = title.replace("/", "_")
  plt.title(title)
  
  epoch = data["Epoch"]
  eps = 0.05
  for ax, name in zip(axes, data):
    ax.plot(epoch, data[name])
    ax.set_title(name)
    ax.set_xlim(np.min(epoch)-eps, 
                np.max(epoch)+eps)
    ax.set_ylim(np.min(data[name])-eps, 
                np.max(data[name])+eps)
    
  plt.tight_layout()
  plt.savefig("../img/{}.pdf".format(title))
  plt.close()
  
def settingsdict(settings):
  sdict = {}
  sdict["name"] = settings[0].strip()
  keys = [
    "Descenter",
    "fx",
    "seed",
    "K ",
    ]
  for line in settings:
    for key in keys:
      if key in line:
        value = line.split(key)[1].strip()
        sdict[key] = value
        
  nl = sdict["fx"].split("[")[-1].split(", 'sig")[0]
  nlsplits = nl.split(",")
  sdict["nl"] = "{}*{}".format(
    nlsplits[0].split("'")[1],
    len(nlsplits))
  
  return sdict

def maaloetonp(infile):
  # Read the whole file into memory.
  with open(infile, "r") as fin:
    lines = fin.readlines()
  
  start = 0
  for i, line in enumerate(lines):
    if "seed" in line:
      seed = line.split("seed ")[1].split(".")[0]
    if "### INITIAL" in line:
      start = i
      
    
  splits = [l.split(";") for l in lines[start:]]
  splits = [s[:6] for s in splits if len(s) == 8]
  settings = {
    "name": "Maaloe",
    "Descenter": "Adam",
    "seed": seed,
    "K ": "1",
    "nl": "rectify",
    }
  data = {}
  for ss in splits:
    for s in ss:
      k,v = s.split("=")
      k = k.strip()
      if k == "epoch":
        k = "Epoch"
      if not k in data:
        data[k] = []
      data[k].append(v.split("%")[0])
  for k, v in data.items():
    data[k] = np.array(map(float,v))
  
  if "test" in data:
    test = data["test"][-1]
    if test < 50:
      print(data["test"][-1], infile)
    
  return settings, data
  
def compareplots(cases, 
                 dirname="compares", 
                 maaloe=False,
                 filtercases=None):
  if not maaloe:
    setdatas = [logtonp(case) for case in cases]
  else:
    setdatas = [maaloetonp(c) for c in cases]
  settings, datas = list(zip(*setdatas))
  
  # Find the fields that are common in all
  # cases.  These are the ones we can plot.
  fields = set.intersection(
    *[set(data.keys()) 
      for data in datas if data.keys()])
  print(fields)
  
  hues = np.linspace(0,1,len(datas),
                     endpoint=False)
  #hues += 0.25
  value = 1.0
  saturation = 0.75
  colors = [
    hsv_to_rgb(np.array([hue,value,saturation]))
    for hue in hues]
  
  # Create two sets of plots -- one versus 
  # epoch (over time) and one versus accuracy.
  # Try to find how accuracy is related to the
  # objectives and the lower bounds.
  plots = {
    "epoch": {
      "data": "Epoch",
      "xscale": "log",
      "xlabel": "Epoch (logarithmic)",
      },
    "accuracy": {
      "data": "test",
      "xlabel": "Test Data Accuracy (percentage)",
      },
    }
  labels = {
    "test": "Test Data Accuracy (percentage)",
    }
  
  for field in fields:
    for ver in plots:
      dkey = plots[ver]["data"]
      
      outname = os.path.join(
        dirname,
        "{}_v{}.pdf".format(field,ver).replace(" ",""))
      print(outname)
      fig = plt.figure(figsize=(8,6))
      lines = {}
      anyplotted = False
      for data, case, color, sdict in zip(
          datas, cases, colors, settings):
        if data.keys():
          plotcase = True
          if filtercases is not None:
            for k, v in filtercases.items():
              if not sdict[k] == v:
                plotcase = False
          if plotcase:
            anyplotted = True
            label = ", ".join([
              "{:>5.2f}%".format(
                np.mean(data["test"][-3:])),
              sdict["name"],
              sdict["K "],
              sdict["seed"],
              sdict["Descenter"],
              sdict["nl"],
              ])
            
            ys = data[field]
            valid = ~np.isnan(ys) & ~np.isinf(ys)
            
            xs = data[dkey][valid]
            ys = ys[valid]
            q99 = np.percentile(ys, 97)
            low = ys < q99
            early = np.arange(len(ys)) < 40
            choose = low | early
            
            lines[case] = plt.plot(
              xs[choose], ys[choose], 
              c=color, label=label)
      
      if anyplotted:
        if "xscale" in plots[ver]:
          plt.gca().set_xscale(plots[ver]["xscale"])
        
        if field in labels:
          plt.ylabel(labels[field])
        else:
          plt.ylabel(field)
        plt.xlabel(plots[ver]["xlabel"])
        plt.axis("tight")
        
        leg = plt.legend(loc="best",
                        prop={"size":10},
                        )
        llines = leg.get_lines()
        plt.setp(llines, linewidth=6.0)
        
        plt.grid("on", which="both")
        plt.tight_layout()
        plt.savefig(outname)
        plt.close()

def usefind(indir):
  #indir = "../../2016_10_22_adgm/log/2016_11_10_sdgm_25epoch_k1/"
  find_output = subprocess.check_output(
    'find {} | grep -i adgm.txt | sort'.format(
      indir), shell=True)
  cases = find_output.decode().split("\n")
  cases = [c for c in cases if c]
  compareplots(cases, dirname="compares/ADGM/Adam",
               filtercases={"Descenter":"Adam"})
  compareplots(cases, dirname="compares/ADGM/Adadelta",
               filtercases={"Descenter":"Adadelta"})
  compareplots(cases, dirname="compares/ADGM")
  
def usels(indir, outdir="compares"):
  find_output = subprocess.check_output(
    'ls {} | grep -i dgm.txt | sort'.format(
      indir), shell=True)
  cases = find_output.decode().split("\n")
  cases = [os.path.join(indir,c) for c in cases if c]
  compareplots(cases, dirname=outdir)
  
def maaloe():
  indir = "/home/kaboo/2016_11_02_maaloe_sdgm/auxiliary-deep-generative-models/output/"
  find_output = subprocess.check_output(
    'find {} | grep -i .log | sort'.format(
      indir), shell=True)
  cases = find_output.decode().split("\n")
  cases = [c for c in cases if c]
  compareplots(cases, dirname="maaloe", maaloe=True)
  
  
if __name__ == "__main__":
  indir = "../log/"
  outdir = "../plots/"
  usels(indir, outdir=outdir)
  
  #maaloe()
