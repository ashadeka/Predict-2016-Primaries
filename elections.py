from sklearn.ensemble import RandomForestClassifier
import sklearn.linear_model as lm
import matplotlib.pylab as plt
import numpy.random as rnd
import pandas as pd
from sklearn.metrics import confusion_matrix

# Get fraction votes per county and latitude and longitude
prim = pd.read_csv("primary_results.csv", index_col=["state_abbreviation", "county"],
                   usecols=["state_abbreviation", "county", "party", "candidate", "fraction_votes"])
coord = pd.read_csv("zip_codes_states.csv", usecols=["state", "county", "latitude", "longitude"])

# Sep primaries into democrats and republicans
prim.index.names = ["state", "county"]
avg_coord = coord.groupby(["state", "county"]).mean()
merged = pd.merge(prim, avg_coord, left_index=True, right_index=True)
dems = merged[merged["party"] == "Democrat"].reset_index(level=("state", "county"))
reps = merged[merged["party"] == "Republican"].reset_index(level=("state", "county"))
merged.to_csv("merged.csv")
# Create a democratic primary votes df
hillary = dems[dems["candidate"] == "Hillary Clinton"].set_index(["state", "county"])
bernie = dems[dems["candidate"] == "Bernie Sanders"].set_index(["state", "county"])
votes = pd.merge(hillary, bernie, suffixes=["hillary", "bernie"], left_index=True, right_index=True)
votes["diff"] = votes["fraction_voteshillary"] / (votes["fraction_voteshillary"] + votes["fraction_votesbernie"])
votes.dropna(inplace = True)



# Create a republican primary votes df
trump = reps[reps["candidate"] == "Donald Trumlp"].set_index(["state", "county"])
cruz = reps[reps["candidate"] == "Ted Cruz"].set_index(["state", "county"])
repsvotes = pd.merge(trump, cruz, suffixes=["trump", "cruz"], left_index=True, right_index=True)
repsvotes["diff"] = repsvotes["fraction_votestrump"] / (repsvotes["fraction_votestrump"] + repsvotes["fraction_votescruz"])
repsvotes.dropna(inplace = True)

# Compute democratic correlation
votes["hillary_win"] = votes["diff"] > 0.5
votes["hillary_win"] = votes["fraction_voteshillary"] > votes["fraction_votesbernie"]
corr = votes.corr()["diff"]


# Compute democratic correlation without New England
votes_adjusted = votes[votes["longitudehillary"] < -74]
votes_adjusted["hillary_win"] = votes_adjusted["diff"] > 0.5
votes_adjusted["hillary_win"] = votes_adjusted["fraction_voteshillary"] > votes_adjusted["fraction_votesbernie"]
corr2 = votes_adjusted.corr()["diff"]



# Compute republican correlation (PIECE-WISE)

repsvotes["trump_win"] = repsvotes["diff"] > 0.5
votes["trump_win"] = repsvotes["fraction_votestrump"] > repsvotes["fraction_votescruz"]
repscorr = repsvotes.corr()["diff"]

repsvotes1 = repsvotes[repsvotes["longitudetrump"]<-108]
repsvotes1 = repsvotes[repsvotes["longitudecruz"]<-108]
repsvotes1["trump_win"] = repsvotes1["diff"] > 0.5
votes["trump_win"] = repsvotes1["fraction_votestrump"] > repsvotes1["fraction_votescruz"]
repscorr1 = repsvotes1.corr()["diff"]

repsvotes2 = repsvotes[repsvotes["longitudetrump"] >-108]
repsvotes2 = repsvotes[repsvotes["longitudecruz"]> -108]
repsvotes2["trump_win"] = repsvotes2["diff"] > 0.5
votes["trump_win"] = repsvotes2["fraction_votestrump"] > repsvotes2["fraction_votescruz"]
repscorr2 = repsvotes2.corr()["diff"]

#Regression Lines

olm = lm.LinearRegression()

# TODO Add labels, title, and style for graphs
# Compute three scatterplots
plt.style.use("ggplot")
olm.fit(votes[["latitudebernie"]].values, votes["diff"].values)
plt.figure()
plt.title("Clinton's Margin of Victory vs Latitude")
plt.xlabel("Latitude")
plt.ylabel("Clinton's Margin of Victory")
plt.scatter(votes["latitudebernie"], votes["diff"])
plt.plot(votes["latitudehillary"],olm.predict(votes[["latitudebernie"]].values))
plt.annotate("corr = %.3f" % corr["latitudehillary"] ,xy=(23,0.2))
plt.savefig("lat-diff.png")


olm.fit(votes[["longitudehillary"]].values, votes["diff"].values)

votes_adjusted = votes[votes["longitudehillary"]< -74]
votes_adjusted = votes[votes["longitudebernie"] < -74]

votes1 = votes[votes["longitudehillary"] > -74]
votes1 = votes[votes["longitudebernie"] > -74]

plt.figure()
plt.title("Clinton's Margin of Victory vs Longitude")
plt.xlabel("Longitude")
plt.ylabel("Clinton's Margin of Victory")
plt.scatter(votes["longitudehillary"], votes["diff"])
plt.plot(votes["longitudehillary"], olm.predict(votes[["longitudehillary"]].values))
plt.scatter(votes1[["longitudehillary"]].values, votes1["diff"].values, color = "red")
olm.fit(votes_adjusted[["longitudehillary"]].values, votes_adjusted["diff"].values)
plt.plot(votes_adjusted["longitudehillary"],olm.predict(votes_adjusted[["longitudehillary"]].values), color = "black")
plt.legend(["New England Regression","non-New England regression", "non-NE","NE"],prop={'size':6}, loc = "lower right")
plt.annotate("corr (no NE) = %.3f" % corr2["longitudehillary"] ,xy=(-130,1))
plt.annotate("corr = %.3f" % corr["longitudehillary"] ,xy=(-130,1.1))

plt.savefig("long-diff.png")


olm.fit(repsvotes[["latitudetrump"]].values, repsvotes["diff"].values)

plt.figure()
plt.title("Trump's Margin of Victory vs Latitude")
plt.xlabel("Latitude")
plt.ylabel("Trump's Margin of Victory")
plt.plot(repsvotes["latitudetrump"],olm.predict(repsvotes[["latitudetrump"]].values))
plt.scatter(repsvotes["latitudecruz"], repsvotes["diff"])
plt.annotate("corr = %.3f" % repscorr["latitudetrump"] ,xy=(42,.05))
plt.savefig("reps-lat-diff.png")


repsvotes1 = repsvotes[repsvotes["longitudetrump"]> - 108]
repsvotes1 = repsvotes[repsvotes["longitudecruz"] > -108]

repsvotes2 = repsvotes[repsvotes["longitudetrump"]< - 108]
repsvotes2= repsvotes[repsvotes["longitudecruz"] < -108]
olm.fit(repsvotes1[["longitudetrump"]].values, repsvotes1["diff"].values)

plt.figure()
plt.title("Trump's Margin of Victory vs Longitude")
plt.xlabel("Longitude")
plt.ylabel("Trump's Margin of Victory")
plt.plot(repsvotes1["longitudetrump"],olm.predict(repsvotes1[["longitudetrump"]].values))
olm.fit(repsvotes2[["longitudetrump"]].values, repsvotes2["diff"].values)
plt.plot(repsvotes2["longitudetrump"],olm.predict(repsvotes2[["longitudetrump"]].values), color = "black")


plt.annotate("corr[1] = %.3f" % repscorr2["longitudetrump"] ,xy=(-90,.04))
plt.annotate("corr[0] = %.3f" % repscorr1["longitudetrump"] ,xy=(-90,.1))
plt.scatter(repsvotes["longitudetrump"], repsvotes["diff"])


plt.savefig("reps-long-diff.png")

# Setup for machine learning
votes2 = votes[['latitudehillary', 'longitudehillary', 'hillary_win']]
selection = rnd.binomial(1, 0.7, size=len(votes2)).astype(bool)
training = votes2[selection]
testing = votes2[~selection]
rfc = RandomForestClassifier()
rfc.fit(training[['latitudehillary', 'longitudehillary']], training['hillary_win'])
demspredicted = rfc.predict(testing[['latitudehillary', 'longitudehillary']])

repsvotes2 = repsvotes[['latitudetrump', 'longitudetrump', 'trump_win']]
repsselection = rnd.binomial(1, 0.7, size=len(repsvotes2)).astype(bool)
repstraining = repsvotes2[repsselection]
repstesting = repsvotes2[~repsselection]
repsrfc = RandomForestClassifier()
repsrfc.fit(repstraining[['latitudetrump', 'longitudetrump']], repstraining['trump_win'])
repspredicted = repsrfc.predict(repstesting[['latitudetrump', 'longitudetrump']])

# Compute accuracy of train and test
train_err = training['hillary_win'] ^ rfc.predict(training[['latitudehillary', 'longitudehillary']])
test_err = testing['hillary_win'] ^ rfc.predict(testing[['latitudehillary', 'longitudehillary']])
train_acc = sum(train_err) / len(train_err)
test_acc = sum(test_err) / len(test_err)

#compute democratic confusion matrix

demscm = confusion_matrix(testing["hillary_win"], demspredicted)
print(demscm)
plt.figure()
plt.imshow(demscm, interpolation='nearest', cmap=plt.cm.Blues)



# TODO Add machine learning for Republicans
reps_train_err = repstraining['trump_win'] ^ repsrfc.predict(repstraining[['latitudetrump', 'longitudetrump']])
reps_test_err = repstesting['trump_win'] ^ repsrfc.predict(repstesting[['latitudetrump', 'longitudetrump']])
reps_train_acc = sum(reps_train_err) / len(reps_train_err)
reps_test_acc = sum(reps_test_err) / len(reps_test_err)

#compute republican confusion matrix
repscm = confusion_matrix(repstesting["trump_win"], repspredicted)
print(repscm)
plt.figure()
plt.imshow(repscm, interpolation='nearest', cmap=plt.cm.Blues)


print("### DEMOCRATS ###")
print("corr of latitude w/ hillary: ", corr[1])
print("corr of longitude w/ hilary: ", corr[2])
print("corr of latitude w/ hillary WITHOUT New England: ", corr2[1])
print("corr of longitude w/ hilary WITHOUT New Enland: ", corr2[2])
print("dems_train_acc: ", (1-train_acc)*100)
print("dems_test_acc: ", (1-test_acc)*100)

print("### REPUBLICANS ###")
print("### First piece ###")

print("corr of latitude w/ Trump: ", repscorr[1])
print("first piece of longitude")
print("corr of longitude w/ Trump: ", repscorr1[2])
print("second piece of longitude")
print("corr of longitude w/ Trump: ", repscorr2[2])

print("reps_train_acc: ", (1-reps_train_acc)*100)
print("reps_test_acc: ", (1-reps_test_acc)*100)
