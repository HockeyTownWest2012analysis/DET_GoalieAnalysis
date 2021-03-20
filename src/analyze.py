import numpy as np
import matplotlib.pyplot as plt
import datetime

#Define Teams Class
class TEAM_STATS(object):
	def __init__(self,team_abbreviations=None,full_names=None,gf60=None,ga60=None):
		self.abbrv = team_abbreviations
		self.teams = full_names
		self.gf60  = gf60
		self.ga60  = ga60

	def get_gf60(self,IN_ABBRV):
		input_team = self.abbrv[IN_ABBRV]
		team_idx   = np.where(self.teams==input_team)[0][0]
		team_gf60  = self.gf60[team_idx]
		return team_gf60

	def get_ga60(self,IN_ABBRV):
		input_team = self.abbrv[IN_ABBRV]
		team_idx   = np.where(self.teams==input_team)[0][0]
		team_ga60  = self.ga60[team_idx]
		return team_ga60

	def get_stats(self,IN_ABBRV):
		team_gf60 = self.get_gf60(IN_ABBRV)
		team_ga60 = self.get_ga60(IN_ABBRV)
		return team_gf60,team_ga60


#Define Goalie Class
class GOALIE_STATS(object):
	def __init__(self,DATE,OPP,GOALIE,GF,GA,TOI,goalie_name='',plot_points=None):
		this_goalie_idx    = np.where(GOALIE==goalie_name)[0]
		self.dates         = DATE[this_goalie_idx]
		self.dates	   = [datetime.datetime.strptime(ts,'%m/%d/%Y') for ts in self.dates]
		self.dates         = [datetime.datetime.strftime(ts,'%m/%d/%Y') for ts in self.dates] #Lazy way to add leading zero in the month
		self.num_games     = len(self.dates)
		self.opponents     = OPP[this_goalie_idx]
		self.goals_for     = GF[this_goalie_idx].astype(float)
		self.goals_against = GA[this_goalie_idx].astype(float)
		self.toi_string    = TOI[this_goalie_idx]
		self.toi           = np.zeros(len(self.toi_string),dtype=float)
		#Convert toi to be entirely as minutes, instead of MM:SS
		for idx in range(len(self.toi_string)):
			split_toi     = self.toi_string[idx].split(":")
			split_min     = float(split_toi[0])
			split_sec     = float(split_toi[1])
			self.toi[idx] = split_min + split_sec/60.

		#Determine goals for/against stats and distribution
		self.avg_goals_for     = np.average(self.goals_for)
		self.std_goals_for     = np.std(self.goals_for)
		self.avg_goals_against = np.average(self.goals_against)
		self.std_goals_against = np.std(self.goals_against)

		self.bins = np.arange(0,11)
		self.goals_for_hist    = np.histogram(self.goals_for,bins=self.bins)[0]
		self.goals_against_hist= np.histogram(self.goals_against,bins=self.bins)[0]

		#Go from PDF to CDF
		self.goals_for_cdf     = np.zeros(len(self.bins)-1,dtype=float)
		self.goals_for_pdf     = self.goals_for_hist.astype(float)/self.num_games

		self.goals_against_cdf = np.zeros(len(self.bins)-1,dtype=float)
		self.goals_against_pdf = self.goals_against_hist.astype(float)/self.num_games
		for idx in range(len(self.bins)-1):
			self.goals_for_cdf[idx]     = np.sum(self.goals_for_pdf[:idx+1])
			self.goals_against_cdf[idx] = np.sum(self.goals_against_pdf[:idx+1])

		self.total_goals_for   = np.sum(self.goals_for)
		self.total_toi         = np.sum(self.toi)
		self.goals_for_per_60  = self.total_goals_for / (self.total_toi / 60.)

		#Save info to correlate date vs idx for plotting timecourse of goals for/against
		if plot_points is not None:
			self.plot_points = np.zeros(len(self.dates),dtype=int)
			for idx in range(len(self.dates)):
				self.plot_points[idx] = plot_points[self.dates[idx]]

	def determine_quality_of_opp(self,TEAM_STATS):
		self.opponent_def_quality = np.zeros(self.num_games,dtype=float)
		self.opponent_off_quality = np.zeros(self.num_games,dtype=float)

		for idx in range(self.num_games):
			self.opponent_def_quality[idx],self.opponent_off_quality[idx] = TEAM_STATS.get_stats(self.opponents[idx])

		#Determine avg opponent qualities
		self.avg_opponent_def_quality = np.average(self.opponent_def_quality)
		self.avg_opponent_off_quality = np.average(self.opponent_off_quality)
		#Weight by TOI, norm'd to 60 min (per total game time)
		self.weighted_opponent_def_quality = np.sum(np.multiply(self.toi/60.,self.opponent_def_quality))/self.num_games
		self.weighted_opponent_off_quality = np.sum(np.multiply(self.toi/60.,self.opponent_off_quality))/self.num_games

#Import Goalie Stats, compiled from hockey-reference.com
DATE,OPP,GOALIE,GF,GA,TOI = np.genfromtxt("RW_Goalies_Stats.txt",dtype=str,unpack=True,skip_header=1,delimiter='\t')
GF = GF.astype(float)
GA = GA.astype(float)

#Dictionary for converting Date to plotting indice
unique_dates = np.unique(DATE)
unique_dates = [datetime.datetime.strptime(ts,"%m/%d/%Y") for ts in unique_dates]
unique_dates.sort()
unique_dates_sorted = [datetime.datetime.strftime(ts,"%m/%d/%Y") for ts in unique_dates]
plot_points = {}
axes_labels = []
for idx in range(len(unique_dates)):
	plot_points[unique_dates_sorted[idx]] = idx
	axes_labels.append(unique_dates_sorted[idx])

#Import Team Stats, from hockey-reference.com
TEAM,GF60,GA60 = np.genfromtxt("Team_Stats.txt",dtype=str,usecols=(1,15,16),delimiter=",",unpack=True,skip_header=1)

#Dictionary to convert abbreviation to full team name
ABBRV = {'TBL':'Tampa Bay Lightning','WSH':'Washington Capitals','VGK':'Vegas Golden Knights','CAR':'Carolina Hurricanes','NYI':'New York Islanders','FLA':'Florida Panthers','EDM':'Edmonton Oilers','TOR':'Toronto Maple Leafs','COL':'Colorado Avalanche','WPG':'Winnipeg Jets','MIN':'Minnesota Wild','PIT':'Pittsburgh Penguins','BOS':'Boston Bruins','STL':'St. Louis Blues','MON':'Montreal Canadiens','VAN':'Vancouver Canucks','CHI':'Chicago Blackhawks','PHI':'Philadelphia Flyers','CGY':'Calgary Flames','CBJ':'Columbus Blue Jackets','LAK':'Los Angeles Kings','ARI':'Arizona Coyotes','NYR':'New York Rangers','NSH':'Nashville Predators','SJS':'San Jose Sharks','DAL':'Dallas Stars','ANA':'Anaheim Ducks','DET':'Detroit Red Wings','NJD':'New Jersey Devils','OTT':'Ottawa Senators','BUF':'Buffalo Sabres'}

#Build Team Classes
team_stats = TEAM_STATS(team_abbreviations=ABBRV,full_names = TEAM, gf60 = GF60.astype(float), ga60 = GA60.astype(float))

#Now, look at each goalie's stats
bernier = GOALIE_STATS(DATE,OPP,GOALIE,GF,GA,TOI,goalie_name='Bernier',plot_points=plot_points)
greiss  = GOALIE_STATS(DATE,OPP,GOALIE,GF,GA,TOI,goalie_name='Greiss',plot_points=plot_points)
pickard = GOALIE_STATS(DATE,OPP,GOALIE,GF,GA,TOI,goalie_name='Pickard',plot_points=plot_points)

#Determine opponent qualities
bernier.determine_quality_of_opp(team_stats)
greiss.determine_quality_of_opp(team_stats)

#Print some stats
print("Goals for / 60:\n=================")
print("  Bernier: "+str(bernier.goals_for_per_60))
print("   Greiss: "+str(greiss.goals_for_per_60))
print("\n\n")

print("Percent of Games with 2 or Fewer Goals:\n=======================================")
print("  Bernier: "+str(bernier.goals_for_cdf[2]))
print("   Greiss: "+str(greiss.goals_for_cdf[2]))
print("\n\n")

gbins = bernier.bins[:-1]

#Plot the distributions all at once (for my analysis)
plt.figure(figsize=(6,6))
plt.subplot(221)
plt.bar(gbins,bernier.goals_for_pdf,color='black',label='Bernier',alpha=0.6)
#plt.axvline(bernier.avg_opponent_def_quality,color='black',linestyle='--')
plt.axvline(bernier.weighted_opponent_def_quality,color='grey',linestyle='--')
plt.bar(gbins,greiss.goals_for_pdf,color='red',label='Greiss',alpha=0.6)
plt.axvline(greiss.weighted_opponent_def_quality,color=(1.0,0.2,0.2),linestyle='--')
#plt.axvline(greiss.avg_opponent_def_quality,color='red',linestyle='--')
plt.xlabel('Goals For')
plt.ylabel('PDF(GF)')
plt.xticks(gbins[::2])
plt.legend(loc=0)
plt.tight_layout()
plt.subplot(222)
plt.plot(gbins,bernier.goals_for_cdf,color='black',label='Bernier')
plt.axvline(bernier.weighted_opponent_def_quality,color='grey',linestyle='--')
plt.plot(gbins,greiss.goals_for_cdf,color='red',label='Greiss')
plt.axvline(greiss.weighted_opponent_def_quality,color=(1.0,0.2,0.2),linestyle='--')
plt.xlabel('Goals For')
plt.ylabel('CDF(GF)')
plt.xticks(gbins[::2])
plt.legend(loc=0)
plt.tight_layout()

plt.subplot(223)
plt.bar(gbins,bernier.goals_against_pdf,color='black',label='Bernier',alpha=0.6)
#plt.axvline(bernier.avg_opponent_def_quality,color='black',linestyle='--')
plt.axvline(bernier.weighted_opponent_off_quality,color='grey',linestyle='--')
plt.bar(gbins,greiss.goals_against_pdf,color='red',label='Greiss',alpha=0.6)
plt.axvline(greiss.weighted_opponent_off_quality,color=(1.0,0.2,0.2),linestyle='--')
#plt.axvline(greiss.avg_opponent_def_quality,color='red',linestyle='--')
plt.xlabel('Goals Against')
plt.ylabel('Prob(GA)')
plt.xticks(gbins[::2])
plt.legend(loc=0)
plt.tight_layout()
plt.subplot(224)
plt.plot(gbins,bernier.goals_against_cdf,color='black',label='Bernier')
plt.axvline(bernier.weighted_opponent_off_quality,color='grey',linestyle='--')
plt.plot(gbins,greiss.goals_against_cdf,color='red',label='Greiss')
plt.axvline(greiss.weighted_opponent_off_quality,color=(1.0,0.2,0.2),linestyle='--')
plt.xlabel('Goals Against')
plt.ylabel('CDF(GA)')
plt.xticks(gbins[::2])
plt.legend(loc=0)
plt.tight_layout()

plt.show()

#Plot the distributions separately, for write-up
plt.figure(figsize=(3.5,3.5))
plt.bar(gbins,bernier.goals_for_pdf,color='black',label='Bernier',alpha=0.6)
#plt.axvline(bernier.avg_opponent_def_quality,color='black',linestyle='--')
plt.axvline(bernier.weighted_opponent_def_quality,color='grey',linestyle='--')
plt.bar(gbins,greiss.goals_for_pdf,color='red',label='Greiss',alpha=0.6)
plt.axvline(greiss.weighted_opponent_def_quality,color=(1.0,0.2,0.2),linestyle='--')
#plt.axvline(greiss.avg_opponent_def_quality,color='red',linestyle='--')
plt.xlabel('Goals For')
plt.ylabel('PDF(GF)')
plt.xticks(gbins[::2])
plt.legend(loc=0)
plt.title("Offensive Support")
plt.tight_layout()
plt.savefig("Offensive_Support_Hist.pdf",format='pdf')

plt.figure(figsize=(3.5,3.5))
plt.plot(gbins,bernier.goals_for_cdf,color='black',label='Bernier')
plt.axvline(bernier.weighted_opponent_def_quality,color='grey',linestyle='--')
plt.plot(gbins,greiss.goals_for_cdf,color='red',label='Greiss')
plt.axvline(greiss.weighted_opponent_def_quality,color=(1.0,0.2,0.2),linestyle='--')
plt.xlabel('Goals For')
plt.ylabel('CDF(GF)')
plt.xticks(gbins[::2])
plt.legend(loc=0)
plt.title("Offensive Support")
plt.tight_layout()
plt.savefig("Offensive_Support_CDF.pdf",format='pdf')

plt.figure(figsize=(3.5,3.5))
plt.bar(gbins,bernier.goals_against_pdf,color='black',label='Bernier',alpha=0.6)
#plt.axvline(bernier.avg_opponent_def_quality,color='black',linestyle='--')
plt.axvline(bernier.weighted_opponent_off_quality,color='grey',linestyle='--')
plt.bar(gbins,greiss.goals_against_pdf,color='red',label='Greiss',alpha=0.6)
plt.axvline(greiss.weighted_opponent_off_quality,color=(1.0,0.2,0.2),linestyle='--')
#plt.axvline(greiss.avg_opponent_def_quality,color='red',linestyle='--')
plt.xlabel('Goals Against')
plt.ylabel('Prob(GA)')
plt.xticks(gbins[::2])
plt.legend(loc=0)
plt.title("Defensive Support")
plt.tight_layout()
plt.savefig("Defensive_Support_Hist.pdf",format='pdf')

plt.figure(figsize=(3.5,3.5))
plt.plot(gbins,bernier.goals_against_cdf,color='black',label='Bernier')
plt.axvline(bernier.weighted_opponent_off_quality,color='grey',linestyle='--')
plt.plot(gbins,greiss.goals_against_cdf,color='red',label='Greiss')
plt.axvline(greiss.weighted_opponent_off_quality,color=(1.0,0.2,0.2),linestyle='--')
plt.xlabel('Goals Against')
plt.ylabel('CDF(GA)')
plt.xticks(gbins[::2])
plt.legend(loc=0)
plt.title("Defensive Support")
plt.tight_layout()
plt.savefig("Defensive_Support_CDF.pdf",format='pdf')


#Comparison on per-game basis
plt.figure(figsize=(3.5,3.5))
plt.plot(bernier.goals_for/(bernier.toi/60.),bernier.opponent_def_quality,color='black',marker='.',linestyle='',label='Bernier')
print(bernier.goals_for/(bernier.toi/60.))
plt.plot(greiss.goals_for/(greiss.toi/60.),greiss.opponent_def_quality,color='red',marker='.',linestyle='',label='Greiss')
plt.plot([0,10],[0,10],linestyle='--',color='grey')
plt.xlabel('DET Goals For/60.')
plt.ylabel('Opponents GA/60')
plt.legend(loc=0)
plt.ylim(-1,10)
plt.xlim(-1,10)
plt.title("Offensive Support")
plt.tight_layout()
plt.savefig("Offensive_Support_Correlation.pdf",format='pdf')

