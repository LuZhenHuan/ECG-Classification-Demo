from django.shortcuts import render
from fetch_profile import fetchPSNProfile
# Create your views here.

def home(request):
    return render(request, 'index.html')

def getProfile(request):
    psn_id=None
    if 'psn_id' in request.GET:
        psn_id = request.GET['psn_id']
    fetch = fetchPSNProfile(psn_id)



    if fetch.isExist():
        level = fetch.getLevel()

        trophy_num = fetch.getTrophy()['trophy_num']
        bronze_trophy = fetch.getTrophy()['bronze_trophy']
        silver_trophy = fetch.getTrophy()['silver_trophy']
        gold_trophy = fetch.getTrophy()['gold_trophy']
        platinum_trophy = fetch.getTrophy()['platinum_trophy']

        avatar_url = fetch.getAvatarUrl()

        last_played = fetch.getLastPlayed()
        return render(request, 'result.html', {'psn_id': psn_id,
                                               'level': level,
                                               'trophy_num': trophy_num,
                                               'bronze_trophy': bronze_trophy,
                                               'silver_trophy': silver_trophy,
                                               'gold_trophy': gold_trophy,
                                               'platinum_trophy': platinum_trophy,
                                               'avatar_url': avatar_url,
                                               'last_played': last_played})
    else:
        return render(request, 'error.html', {'content': 'PSN ID "'+str(psn_id)+'" does not exist'})
