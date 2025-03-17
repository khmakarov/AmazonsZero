#include "action_index.h"
void load_actions()
{
	int count = 0;
	std::ifstream f("E:/VSCPython/AmazonsZero/data/actions/amazons_actions.bin", std::ios::binary);

	while (true)
	{
		std::tuple<uint8_t, uint8_t, uint8_t> action;
		f.read(reinterpret_cast<char *>(&action), sizeof(std::tuple<uint8_t, uint8_t, uint8_t>));
		if (f.eof())
			break;

		action_list[count] = action;
		action_map[(static_cast<size_t>(std::get<0>(action)) << 16) | (static_cast<size_t>(std::get<1>(action)) << 8) | static_cast<size_t>(std::get<2>(action))] = count++;
	}
}

std::pair<std::array<bool, TOTAL_ACTIONS>, std::array<int, 0x800>> generate_mask(const std::pair<std::array<std::tuple<uint8_t, uint8_t, uint8_t>, 0x800>, int> &legal_actions)
{
	auto [actions, counts] = legal_actions;
	std::array<bool, TOTAL_ACTIONS> mask{};
	std::array<int, 0x800> mask_index{counts};
	int count = 0;
	for (const auto &action : actions)
	{
		if (count == counts)
			break;
		if (auto it = action_map.find((static_cast<size_t>(std::get<0>(action)) << 16) | (static_cast<size_t>(std::get<1>(action)) << 8) | static_cast<size_t>(std::get<2>(action))); it != action_map.end())
		{
			mask[it->second] = true;
			mask_index[++count] = static_cast<int>(it->second);
		}
	}
	return {mask, mask_index};
}