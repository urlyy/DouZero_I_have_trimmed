from env.game import GameEnv
from evaluation.rlcard_agent import RLCardAgent
from evaluation.random_agent import RandomAgent
from evaluation.deep_agent import DeepAgent

cards_info = dict()
cards_info['landlord'] =       [3, 3, 3, 4, 4,  5,  5,  5,  6,  7,  7,  8,  9, 11, 12, 12, 13, 13, 17, 20]
cards_info['landlord_up'] =    [3, 4, 5, 6, 6, 10, 10, 10, 11, 12, 13, 13, 14, 14, 17, 17, 30]
cards_info['landlord_down'] =  [4, 6, 7, 7, 8,  8,  8,  9,  9,  9, 10, 11, 11, 12, 14, 14, 17]
cards_info['three_landlord_cards'] = [4, 5, 12]

# 不同模型的玩家
def load_card_play_models():
    card_play_model_path_dict = {'landlord': 'baselines/douzero_ADP/landlord.ckpt',
                                 'landlord_up': 'baselines/sl/landlord_up.ckpt',
                                 'landlord_down': 'baselines/sl/landlord_down.ckpt'}
    players = {}
    for position in ['landlord', 'landlord_up', 'landlord_down']:
        if card_play_model_path_dict[position] == 'rlcard':
            players[position] = RLCardAgent(position)
        elif card_play_model_path_dict[position] == 'random':
            players[position] = RandomAgent()
        else:
            players[position] = DeepAgent(position, card_play_model_path_dict[position])
    return players

def main():
    players = load_card_play_models()
    env = GameEnv(players)
    env.card_play_init(cards_info)
    while not env.game_over:
        env.step()
    print("赢家:",env.winner)
    env.reset()

main()