from CRBot import CRBot

if __name__ == '__main__':
    bot = CRBot()
    episodes = 1          # start with 1 for the first test run
    bot.play(episodes=episodes, load=True, save=True)