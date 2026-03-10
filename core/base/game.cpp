#include "game.h"
std::array<MoveAction, TOTAL_ACTIONS> action_list;
std::unordered_map<size_t, size_t> action_map;
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
int thresold = 0;
int start, current, turnID;
int currBotColor = WHITE;
double C_UCT = 0.01;
double w1[56] = {0, 0, 0.074938141, 2.113308430, 1.829644322, 1.657095432, 1.844633341, 1.709531188, 1.947853684, 1.597983241, 1.290230632, 1.167080998, 0.580433190, 0.053694796, -0.011891359, 0.527658403, 0.975054741, 1.056464434, 0.872889936, 0.678786278, 0.323303133, 0.346939683, 0.260094881, 0.264616042, 0.246926725, 0.189162090, 0.141057417, 0.105003797, 0.100303024, 0.094548225, 0.075991668, 0.056678012, 0.052921258, 0.046266042, 0.047984783, 0.029863806, 0.046501521, 0.035372451, 0.036930695, 0.026028842, 0.023727236, 0.008815981, 0.005980607, 0.009275969, -0.003742765, -0.009044983, -0.010549444, -0.032312561, -0.012371800, -0.037411232, -0.032170087, -0.012665736, -0.025661280, 0.030489521, 0.038260937, -0.033962481};
double w2[56] = {0, 0, -0.071123712, 1.850196242, 1.909757733, 1.786974669, 1.441033125, 1.831614137, 1.863318324, 1.549122095, 0.929738343, 0.985552371, 0.519130468, 0.652193844, 0.677611768, 0.715636075, -0.022837769, 0.100610048, -0.172568828, -0.087590173, 0.244123191, 0.132074401, 0.141869083, 0.097190522, 0.033922136, 0.084543742, 0.094968185, 0.085143305, 0.066933967, 0.070909552, 0.083291963, 0.085179411, 0.069383688, 0.058226194, 0.060640641, 0.065368392, 0.054329056, 0.054176800, 0.042966656, 0.044218585, 0.038709428, 0.031496748, 0.023774918, 0.015536518, 0.010674691, 0.005355561, 0.008374230, 0.010649841, 0.012298613, 0.012731116, 0.006807683, 0.003942049, 0.000137917, -0.014996326, -0.007956794, -0.035135169};
double w3[56] = {0, 0, 0.285573512, 1.067316651, -0.211924806, 0.680028081, 0.517808735, 0.304819107, -0.360255659, 0.283211440, -0.118919544, -0.320547521, -0.164136052, -0.199558660, -0.158322647, -0.134399980, -0.166122779, -0.147537082, -0.119008608, -0.115313880, -0.088800281, -0.144199789, -0.093452334, -0.098844223, -0.061869688, -0.042582039, -0.026260521, -0.000397748, -0.004830216, -0.015821418, 0.015736405, 0.018657653, 0.027041025, 0.018133903, 0.027038572, 0.051898617, 0.024094300, 0.032939505, 0.027083304, 0.032979585, 0.024715593, 0.048389383, 0.051429220, 0.049864184, 0.065294459, 0.079495110, 0.070997894, 0.100243673, 0.052135076, 0.094104849, 0.087983906, 0.061890475, 0.079814047, -0.026632605, -0.043000557, 0.082739472};
double w4[56] = {0, 0, 0.061070204, 2.073594332, 1.803056121, 1.969837666, 1.993263960, 1.986328840, 1.706978559, 1.858220458, 1.816977739, 1.320441842, 1.554476857, 1.503799796, 1.165330052, 0.306896836, 0.428119272, 0.127804548, 0.320249408, 0.250915736, 0.208399042, 0.209280223, 0.189255610, 0.176422641, 0.158465251, 0.123073861, 0.114771605, 0.107638910, 0.104587719, 0.094095841, 0.077729441, 0.085181594, 0.088369705, 0.096417032, 0.086386517, 0.084225371, 0.094143234, 0.101544440, 0.116312958, 0.118276447, 0.126188114, 0.131197393, 0.142377511, 0.138548672, 0.131359026, 0.133364707, 0.122829571, 0.130566165, 0.130473971, 0.133546650, 0.111887053, 0.097306497, 0.095673442, 0.109085456, 0.139188647, 0.08377216};
double w5[56] = {0, 0, 0.107754663, -0.528144240, -1.379346251, -1.006276369, -0.825565398, -0.430609971, -0.473824382, -0.060512420, -0.942406714, -0.967272699, -0.992075503, -1.073423266, -0.858994186, -0.838604629, -0.738691866, -0.771107376, -0.667914927, -0.620147049, -0.567148566, -0.658371270, -0.578625798, -0.549303949, -0.488953888, -0.356694996, -0.272865117, -0.213500351, -0.175519645, -0.133494422, -0.070355214, -0.038640279, -0.043525659, -0.037141897, -0.031321511, -0.023366986, -0.021908367, -0.013590249, -0.019244174, -0.014974004, -0.009036653, -0.018890575, -0.004873865, -0.017690081, -0.012989196, -0.020389291, -0.031792857, -0.031873308, -0.028690575, -0.024877874, -0.033742540, -0.026430298, -0.027800240, -0.030750951, -0.021546466, -0.037860770};
double w6[56] = {0, 0, -4.567995548, -3.716784000, -3.381752491, -2.938109636, -3.290686846, -2.702058077, -2.855453491, -2.595350504, -2.005551338, -1.637469649, -1.272953510, -1.047218084, -0.879562616, -0.746166408, -0.717715025, -0.592138231, -0.504874051, -0.366238832, -0.404548317, -0.305050164, -0.272765279, -0.249806121, -0.137687027, -0.254965365, -0.089192919, -0.171341106, -0.076427318, -0.220489189, -0.091823421, -0.180632398, -0.106192604, -0.145687878, -0.073370934, -0.149500415, -0.069663063, -0.189439490, -0.027017180, -0.123841494, -0.011969413, -0.017853998, 0.068564072, 0.002616666, 0.172109649, 0.157967970, 0.158981338, 0.351862490, 0.168605193, 0.400329232, 0.361347109, 0.220473915, 0.434309691, -0.030947627, -0.074839503, 0.44456172};
Board board;
Board::Board()
{
    board_[0][2] = BLACK;
    board_[2][0] = BLACK;
    board_[5][0] = BLACK;
    board_[7][2] = BLACK;
    board_[0][5] = WHITE;
    board_[2][7] = WHITE;
    board_[5][7] = WHITE;
    board_[7][5] = WHITE;
}
bool Board::isValid_Map(int x, int y) const
{
    return x >= 0 && x < N && y >= 0 && y < N;
}
bool Board::isBlank(int x, int y) const
{
    return (board_[x][y] == 0) && isValid_Map(x, y);
}
bool Board::isBlock(int x, int y) const
{
    return board_[x][y] == BLOCK;
}
bool Board::canDo(int x, int y, int nx, int ny) const
{
    if (!isValid_Map(nx, ny) || board_[nx][ny])
        return false;
    return true;
}
int Board::getPiece(int x, int y)
{
    return board_[x][y];
}
void Board::clear(int x, int y)
{
    temp = board_[x][y];
    board_[x][y] = EMPTY;
}
void Board::restore(int x, int y, int nx, int ny, int bx, int by)
{
    if (nx == -1)
    {
        board_[x][y] = temp;
        temp = 0;
    }
    else
    {
        if (x == bx && y == by)
        {
            board_[x][y] = board_[nx][ny];
            board_[nx][ny] = EMPTY;
        }
        else
        {
            board_[x][y] = board_[nx][ny];
            board_[nx][ny] = EMPTY;
            board_[bx][by] = EMPTY;
        }
    }
}
void Board::movePiece(int x, int y, int nx, int ny)
{
    board_[nx][ny] = board_[x][y];
    board_[x][y] = EMPTY;
}
void Board::placeBlock(int x, int y)
{
    board_[x][y] = BLOCK;
}
void Board::print()
{
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            std::cout << board_[i][j];
        }
        std::cout << std::endl;
    }
}
void Board::clear_all()
{
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            board_[i][j] = 0;
        }
    }
    board_[0][2] = BLACK;
    board_[2][0] = BLACK;
    board_[5][0] = BLACK;
    board_[7][2] = BLACK;
    board_[0][5] = WHITE;
    board_[2][7] = WHITE;
    board_[5][7] = WHITE;
    board_[7][5] = WHITE;
}
Node::Node(int player) : player(player)
{
    visits = 0;
    numberOfVisits = 0;
    value = 0.0;
    cut = false;
    parent = nullptr;
}

Evaluate::Evaluate(Node *node) : node(node) {}
void Evaluate::init()
{
    memset(mobValues, 0, sizeof(mobValues));
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (board.isBlank(i, j))
                umap |= (1ull << (i * 8 + j));
            else if (board.getPiece(i, j) == node->player)
                p1 |= 1ull << (i * 8 + j);
            else if (board.getPiece(i, j) == -node->player)
                p2 |= 1ull << (i * 8 + j);
        }
    }
}
uint64_t Evaluate::shiftMask(uint64_t a, int shift, uint64_t num, uint64_t direction, uint64_t can)
{
    uint64_t result = a;
    for (int i = 0; i < 7; i++)
    {
        result |= (shift > 0 ? result << num : result >> num) & direction & can;
    }
    return result;
}
uint64_t Evaluate::applyShifts(uint64_t temp, uint64_t mask, const int shiftDirections[], const uint64_t shiftAmounts[], const uint64_t directions[])
{
    uint64_t result = temp;
    for (int i = 0; i < 8; i++)
    {
        result |= shiftMask(temp, shiftDirections[i], shiftAmounts[i], directions[i], mask);
    }
    return result;
}
void Evaluate::queenBFS()
{
    q1[0] |= p1, q2[0] |= p2;
    int shiftDirections[8] = {-1, 1, -1, 1, -1, 1, -1, 1};
    uint64_t shiftAmounts[8] = {1, 1, 8, 8, 9, 9, 7, 7};
    uint64_t directions[8] = {West, East, North, South, NW, SE, NE, SW};
    do
    {
        ++qcnt1;
        uint64_t temp = q1[qcnt1 - 1];
        q1[qcnt1] = applyShifts(temp, umap | p1, shiftDirections, shiftAmounts, directions);
    } while (q1[qcnt1] != q1[qcnt1 - 1]);
    for (int x = qcnt1 - 1; x >= 1; --x)
        q1[x] ^= q1[x - 1];
    do
    {
        ++qcnt2;
        uint64_t temp = q2[qcnt2 - 1];
        q2[qcnt2] = applyShifts(temp, umap | p2, shiftDirections, shiftAmounts, directions);
    } while (q2[qcnt2] != q2[qcnt2 - 1]);
    for (int x = qcnt2 - 1; x >= 1; --x)
        q2[x] ^= q2[x - 1];
}
void Evaluate::kingBFS()
{
    k1[0] |= p1, k2[0] |= p2;
    do
    {
        ++kcnt1;
        uint64_t a = k1[kcnt1 - 1];
        a |= (a >> 1 & West) | (a << 1 & East) | (a >> 8 & North) | (a << 8 & South) | (a >> 9 & NW) | (a << 9 & SE) | (a >> 7 & NE) | (a << 7 & SW);
        k1[kcnt1] = a & (umap | p1);
    } while (k1[kcnt1] != k1[kcnt1 - 1]);
    for (int x = kcnt1 - 1; x >= 1; --x)
        k1[x] ^= k1[x - 1];
    do
    {
        ++kcnt2;
        uint64_t a = k2[kcnt2 - 1];
        a |= (a >> 1 & West) | (a << 1 & East) | (a >> 8 & North) | (a << 8 & South) | (a >> 9 & NW) | (a << 9 & SE) | (a >> 7 & NE) | (a << 7 & SW);
        k2[kcnt2] = a & (umap | p2);
    } while (k2[kcnt2] != k2[kcnt2 - 1]);
    for (int x = kcnt2 - 1; x >= 1; --x)
        k2[x] ^= k2[x - 1];
}
void Evaluate::pretreatment()
{
    queenBFS();
    kingBFS();
    maxq = std::max(qcnt1, qcnt2);
    maxk = std::max(kcnt1, kcnt2);
    for (int i = 1; i < std::min(qcnt1, qcnt2); ++i)
        uq[i] |= q1[i] | q2[i];
    for (int i = std::min(qcnt1, qcnt2); i < maxq; ++i)
    {
        if (qcnt1 < qcnt2)
            uq[i] |= q2[i];
        else
            uq[i] |= q1[i];
    }
    for (int i = 2; i <= maxq; ++i)
        uq[i] |= uq[i - 1];
    for (int i = 1; i < std::min(kcnt1, kcnt2); ++i)
        uk[i] |= k1[i] | k2[i];
    for (int i = std::min(kcnt1, kcnt2); i < maxk; ++i)
    {
        if (kcnt1 < kcnt2)
            uk[i] |= k2[i];
        else
            uk[i] |= k1[i];
    }
    for (int i = 2; i <= maxk; ++i)
        uk[i] |= uk[i - 1];
}
std::pair<std::pair<double, double>, std::pair<double, double>> Evaluate::computeParameter()
{
    double w1 = 0.0, w2 = 0.0, w3 = 0.0, w4 = 0.0;
    for (int i = 2; i < std::min(qcnt1, qcnt2); ++i)
    {
        w1 += cntParameter[i - 1] * __builtin_popcountll(uq[i - 1] & q1[i]);
        w1 -= cntParameter[i - 1] * __builtin_popcountll(uq[i - 1] & q2[i]);
    }
    for (int i = std::min(qcnt1, qcnt2); (qcnt1 != qcnt2) && (i < maxq); ++i)
    {
        if (qcnt1 < qcnt2)
            w1 -= cntParameter[i - 1] * __builtin_popcountll(uq[i - 1] & q2[i]);
        else
            w1 += cntParameter[i - 1] * __builtin_popcountll(uq[i - 1] & q1[i]);
    }
    w1 += __builtin_popcountll(uq[maxq] & (umap ^ q1[qcnt1])) - __builtin_popcountll(uq[maxq] & (umap ^ q2[qcnt2])) + 0.3 * __builtin_popcountll(q1[1] & q2[1]);
    for (int i = 2; i < std::min(kcnt1, kcnt2); ++i)
    {
        w2 += cntParameter[i - 1] * __builtin_popcountll(uk[i - 1] & k1[i]);
        w2 -= cntParameter[i - 1] * __builtin_popcountll(uk[i - 1] & k2[i]);
    }
    for (int i = std::min(kcnt1, kcnt2); (kcnt1 != kcnt2) && (i < maxk); ++i)
    { // 数组可能越界
        if (kcnt1 < kcnt2)
            w2 -= cntParameter[i - 1] * __builtin_popcountll(uk[i - 1] & k2[i]);
        else
            w2 += cntParameter[i - 1] * __builtin_popcountll(uk[i - 1] & k1[i]);
    }
    w2 += __builtin_popcountll(uk[maxk] & (umap ^ k1[kcnt1])) - __builtin_popcountll(uk[maxk] & (umap ^ k2[kcnt2])) + 0.3 * __builtin_popcountll(k1[1] & k2[1]);
    for (int i = 1; i <= 6 && i < qcnt1; ++i)
        w3 -= __builtin_popcountll(q1[i]) * depthParameter[i];
    for (int i = 1; i <= 6 && i < qcnt2; ++i)
        w3 += __builtin_popcountll(q2[i]) * depthParameter[i];
    uint64_t initial1 = k1[kcnt1], initial2 = k2[kcnt2];
    uint64_t intersect = initial1 & initial2;
    w4 += __builtin_popcountll(initial2 & ~initial1) - __builtin_popcountll(initial1 & ~initial2);
    for (int i = 1; i < kcnt1; ++i)
        w4 += __builtin_popcountll(intersect & k1[i]) * i / 6.0;
    for (int i = 1; i < kcnt2; ++i)
        w4 -= __builtin_popcountll(intersect & k2[i]) * i / 6.0;
    return {{w1, w2}, {w3, w4}};
}
double Evaluate::computeBlankValue()
{
    double w1 = 0, w2 = 0;
    for (blank = umap; blank; blank &= blank - 1)
    {
        int bit = __builtin_ctzll(blank);
        uint64_t ubit = 1ull << bit;
        ubit |= (ubit >> 1 & West) | (ubit << 1 & East) | (ubit >> 8 & North) | (ubit << 8 & South) | (ubit >> 9 & NW) | (ubit << 9 & SE) | (ubit >> 7 & NE) | (ubit << 7 & SW);
        int cnt = __builtin_popcountll(ubit & umap);
        int row = bit / 8, column = bit % 8;
        mobValues[row][column] = cnt - 1;
    }
    for (uint64_t piece = p1; piece; piece &= piece - 1)
    {
        int bit = __builtin_ctzll(piece), cnt = 0;
        uint64_t upiece = 1ull << bit;
        uint64_t a[40] = {0ull};
        a[0] |= upiece;
        do
        {
            ++cnt;
            uint64_t result = a[cnt - 1];
            result |= (result >> 1 & West) | (result << 1 & East) | (result >> 8 & North) | (result << 8 & South) | (result >> 9 & NW) | (result << 9 & SE) | (result >> 7 & NE) | (result << 7 & SW);
            a[cnt] = (result & umap) ^ upiece;
        } while (a[cnt] != a[cnt - 1]);
        for (int x = cnt - 1; x >= 1; --x)
            a[x] ^= a[x - 1];
        for (int i = 1; i < cnt; ++i)
        {
            for (; a[i]; a[i] &= a[i] - 1)
            {
                // to_binary(a[i]);
                int bit = __builtin_ctzll(a[i]);
                int row = bit / 8, column = bit % 8;
                w1 += mobValues[row][column] * cntParameter[i] / i;
            }
        }
    }
    for (uint64_t piece = p2; piece; piece &= piece - 1)
    {
        int bit = __builtin_ctzll(piece), cnt = 0;
        uint64_t upiece = 1ull << bit;
        uint64_t a[40] = {0ull};
        a[0] |= upiece;
        do
        {
            ++cnt;
            uint64_t result = a[cnt - 1];
            result |= (result >> 1 & West) | (result << 1 & East) | (result >> 8 & North) | (result << 8 & South) | (result >> 9 & NW) | (result << 9 & SE) | (result >> 7 & NE) | (result << 7 & SW);
            a[cnt] = (result & umap) ^ upiece;
        } while (a[cnt] != a[cnt - 1]);
        for (int x = cnt - 1; x >= 1; --x)
            a[x] ^= a[x - 1];
        for (int i = 1; i < cnt; ++i)
        {
            for (; a[i]; a[i] &= a[i] - 1)
            {
                int bit = __builtin_ctzll(a[i]);
                int row = bit / 8, column = bit % 8;
                w2 += mobValues[row][column] * cntParameter[i] / i;
            }
        }
    }
    return w1 - w2;
}
double Evaluate::getEvaluateValues()
{
    init();
    int blankCnts = __builtin_popcountll(umap);
    double p1 = w1[blankCnts], p2 = w2[blankCnts], p3 = w3[blankCnts], p4 = w4[blankCnts], p5 = w5[blankCnts], p6 = w6[blankCnts];
    pretreatment();
    auto parameter = computeParameter();
    double a = parameter.first.first;
    double b = parameter.first.second;
    double c = parameter.second.first;
    double d = parameter.second.second;
    double e = 0.1 * computeBlankValue();
    double v = a * p1 + b * p2 + c * p3 + d * p4 + e * p5 + p6;
    return 1.0 - (1 / (1 + exp(-v)));
}

MCTS::MCTS(Node *node) : root(node) {} // 初始化
void MCTS::runSearch()
{ // 启动蒙特卡洛树搜索
    if (!root->legalMove.size())
    {
        bool expandedNode = expandMove(*root);
        if (expandedNode)
        {
            if (root->player == currBotColor)
                backpropagate(root, 1.0);
            else
                backpropagate(root, 0.0);
        }
        else
        {
            Node *child = expandNode(*root);
            board.movePiece(child->action.startX, child->action.startY, child->action.endX, child->action.endY);
            board.placeBlock(child->action.barrierX, child->action.barrierY);
            Evaluate eva(child);
            double v = eva.getEvaluateValues(); // 评估
            board.restore(child->action.startX, child->action.startY, child->action.endX, child->action.endY, child->action.barrierX, child->action.barrierY);
            if (child->player == currBotColor)
                backpropagate(child, v);
            else
                backpropagate(child, 1.0 - v);
        }
    }
    else
    {
        if (root->numberOfVisits < (int)root->legalMove.size())
        {
            Node *child = expandNode(*root);
            board.movePiece(child->action.startX, child->action.startY, child->action.endX, child->action.endY);
            board.placeBlock(child->action.barrierX, child->action.barrierY);
            Evaluate eva(child);
            double v = eva.getEvaluateValues(); // 评估
            if (turnID < 8)
            {
                umoveB |= (1ull << (child->action.startX * 8 + child->action.startY));
                umoveE |= (1ull << (child->action.endX * 8 + child->action.endY));
                if (currBotColor == 1)
                {
                    if (__builtin_popcountll(umoveB & border) && __builtin_popcountll(umoveE & ~border))
                        v *= 1.25;
                    else if (__builtin_popcountll(umoveE & border))
                        v *= 0.9;
                }
                else
                {
                    if (__builtin_popcountll(umoveB & border) && __builtin_popcountll(umoveE & ~border))
                        v *= 1.05;
                }
            }
            board.restore(child->action.startX, child->action.startY, child->action.endX, child->action.endY, child->action.barrierX, child->action.barrierY);
            if (child->player == currBotColor)
                backpropagate(child, v);
            else
                backpropagate(child, 1.0 - v);
        }
        else
        {
            if (!root->cut && root->children.size() > 12)
            {
                root->cut = true;
                if (root->player == -currBotColor)
                {
                    stable_sort(root->children.begin(), root->children.end(), [&](Node *a, Node *b)
                                { return a->value > b->value; });
                    root->children.resize(12);
                }
                else
                {
                    stable_sort(root->children.begin(), root->children.end(), [&](Node *a, Node *b)
                                { return a->value < b->value; });
                    root->children.resize(12);
                }
            }
            Node &nextNode = *select(root);
            board.movePiece(nextNode.action.startX, nextNode.action.startY, nextNode.action.endX, nextNode.action.endY);
            board.placeBlock(nextNode.action.barrierX, nextNode.action.barrierY);
            MCTS nextmcts(&nextNode);
            nextmcts.runSearch();
            board.restore(nextNode.action.startX, nextNode.action.startY, nextNode.action.endX, nextNode.action.endY, nextNode.action.barrierX, nextNode.action.barrierY);
        }
    }
}
std::pair<std::vector<int>, std::vector<double>> MCTS::ans()
{
    std::stable_sort(root->children.begin(), root->children.end(),
                     [](Node *a, Node *b)
                     { return a->value > b->value; });
    size_t num_top = std::min(root->children.size(), static_cast<size_t>(10));
    std::vector<Node *> top_children(root->children.begin(), root->children.begin() + num_top);

    int total_visits = 0;
    for (const auto &child : top_children)
        total_visits += child->visits;

    // 生成动作索引和概率分布
    std::vector<int> action_ints;
    std::vector<double> probs;
    action_ints.reserve(num_top);
    probs.reserve(num_top);

    for (const auto &child : top_children)
    {
        // 计算action_sum
        size_t action_sum = static_cast<size_t>(
            (child->action.startY * 8 + child->action.startX) << 16 |
            ((child->action.endY * 8 + child->action.endX) << 8) |
            (child->action.barrierY * 8 + child->action.barrierX));

        // 获取动作索引
        action_ints.push_back(action_map.at(action_sum));

        // 计算概率
        probs.push_back(static_cast<double>(child->visits) / total_visits);
    }

    return {action_ints, probs};
}

Node *MCTS::select(Node *node)
{
    Node *bestChild = node;
    double bestUCT = -1.0;
    for (const auto &child : node->children)
    {
        double uct = computeUCT(child);
        if (uct > bestUCT)
        {
            bestUCT = uct;
            bestChild = child;
        }
    }
    return bestChild;
}
bool MCTS::expandMove(Node &node)
{
    node.legalMove.reserve(500);
    node.children.reserve(500);
    coordinates Piece[4]{};
    int count = 0;
    bool isFinished = false;
    for (int i = 0; i < N && !isFinished; ++i)
    {
        for (int j = 0; j < N && !isFinished; ++j)
        {
            if (board.getPiece(i, j) == -node.player)
            {
                Piece[count++] = {i, j};
                if (count == 4)
                    isFinished = true;
            }
        }
    }
    for (const auto &piece : Piece)
    {
        int x = piece.x, y = piece.y;
        for (int dir = 0; dir < N; ++dir)
        {
            for (int nx = x + dx[dir], ny = y + dy[dir], tx = x, ty = y; board.canDo(tx, ty, nx, ny); tx = nx, ty = ny, nx += dx[dir], ny += dy[dir])
            {
                board.clear(x, y);
                for (int tdir = 0; tdir < N; ++tdir)
                {
                    for (int bx = nx + dx[tdir], by = ny + dy[tdir], tnx = nx, tny = ny; board.canDo(tnx, tny, bx, by); tnx = bx, tny = by, bx += dx[tdir], by += dy[tdir])
                    {
                        node.legalMove.push_back({x, y, nx, ny, bx, by});
                    }
                }
                board.restore(x, y, -1, -1, -1, -1);
            }
        }
    }
    return node.legalMove.empty();
}
Node *MCTS::expandNode(Node &node)
{
    int luckyNode = rand() % (node.legalMove.size() - node.numberOfVisits);
    Node *child = new Node(-node.player);
    child->action = node.legalMove[luckyNode];
    child->parent = &node;
    node.children.push_back(child);
    std::swap(node.legalMove[luckyNode], node.legalMove[node.legalMove.size() - node.numberOfVisits - 1]);
    node.numberOfVisits++; // 判断是否还要扩展当前节点
    return child;
}
void MCTS::backpropagate(Node *node, double result)
{
    Node *currentNode = node;
    while (currentNode != nullptr)
    {
        currentNode->visits++;
        currentNode->value += result;
        currentNode = currentNode->parent;
    }
}
double MCTS::computeUCT(const Node *node) const
{
    double exploitation = node->value / node->visits;                            // 平均价值
    double exploration = C_UCT * sqrt(log(node->parent->visits) / node->visits); // 扩展值
    if (node->player == currBotColor)
        return exploitation + exploration;
    else
        return 1.0 - exploitation + exploration;
}

std::pair<std::vector<int>, std::vector<double>> Game::step(int tID, py::array_t<int> action_idx)
{
    srand(static_cast<unsigned>(time(0)));
    int x0, y0, x1, y1, x2, y2;
    currBotColor = WHITE; // 先假设自己是白方
    auto buf = action_idx.unchecked<1>();
    turnID = tID;
    for (int i = 0; i < turnID; i++)
    {
        auto [a, d, c] = action_list[buf(i)];
        int f = a, t = d, b = c;
        x0 = f % 8, y0 = f / 8, x1 = t % 8, y1 = t / 8, x2 = b % 8, y2 = b / 8;
        board.movePiece(x0, y0, x1, y1);
        board.placeBlock(x2, y2);
    }

    start = clock();
    C_UCT = 1.35;
    thresold = 0.97 * (double)CLOCKS_PER_SEC;
    Node root(-currBotColor); // 以对手为根节点建树
    MCTS mcts(&root);
    while (1)
    {
        mcts.runSearch();
        current = clock();
        if (current - start > thresold)
            break;
    }
    board.clear_all();
    return mcts.ans();
}