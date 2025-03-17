#include "action_index.h"
void load_actions()
{
	int count = 0;
	std::ifstream f("/home/khmakarov/AmazonsZero/data/actions/amazons_actions.bin", std::ios::binary);
	while (true)
	{
		MoveAction action;
		f.read(reinterpret_cast<char *>(&action), sizeof(MoveAction));
		if (f.eof())
			break;

		action_list[count] = action;
		action_map[static_cast<size_t>(std::get<0>(action)) << 16 | (static_cast<size_t>(std::get<1>(action)) << 8) | static_cast<size_t>(std::get<2>(action))] = count++;
	}
}

std::pair<std::array<bool, TOTAL_ACTIONS>, std::array<int, POSSIBLE_ACTIONS>> generate_mask(const std::pair<std::array<MoveAction, POSSIBLE_ACTIONS>, int> &legal_actions)
{
	int count = 0;
	auto [actions, counts] = legal_actions;
	std::array<bool, TOTAL_ACTIONS> mask{};
	std::array<int, POSSIBLE_ACTIONS> mask_index{counts};
	for (const auto &action : actions)
	{
		if (count == counts)
			break;
		if (auto it = action_map.find(static_cast<size_t>(std::get<0>(action)) << 16 | (static_cast<size_t>(std::get<1>(action)) << 8) | static_cast<size_t>(std::get<2>(action))); it != action_map.end())
		{
			mask[it->second] = true;
			mask_index[++count] = static_cast<int>(it->second);
		}
	}
	return {mask, mask_index};
}