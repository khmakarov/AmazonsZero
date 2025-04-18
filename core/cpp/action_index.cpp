#include "action_index.h"
void load_actions()
{
	int count = 0;
	std::ifstream f("/home/khmakarov/AmazonsZero/data/action_space/amazons_actions.bin", std::ios::binary);
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

py::array_t<int> generate_mask_np(const std::pair<std::array<MoveAction, POSSIBLE_ACTIONS>, int> &legal_actions)
{
	int count = 0;
	std::vector<int> data(POSSIBLE_ACTIONS, -1);
	py::array_t<int> mask_index(POSSIBLE_ACTIONS, data.data());
	auto buf = mask_index.mutable_unchecked<1>();
	auto [actions, counts] = legal_actions;
	buf[0] = counts;
	for (const auto &action : actions)
	{
		if (count == counts)
			break;
		auto action_sum = static_cast<size_t>(std::get<0>(action)) << 16 | (static_cast<size_t>(std::get<1>(action)) << 8) | static_cast<size_t>(std::get<2>(action));
		if (auto it = action_map.find(action_sum); it != action_map.end())
			buf[++count] = static_cast<int>(it->second);
	}
	return mask_index;
}