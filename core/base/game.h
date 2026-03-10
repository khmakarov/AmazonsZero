#ifndef GAME_H
#define GAME_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <array>
#include <cstdint>
#include <fstream>
#include <functional>
#include <immintrin.h>
#include <omp.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <tuple>
#include <unordered_map>
#include <xxhash.h>
namespace py = pybind11;
using MoveAction = std::tuple<uint8_t, uint8_t, uint8_t>;
using ChildState = std::tuple<py::array_t<size_t>, py::array_t<int8_t>, py::array_t<int>>;
constexpr size_t TOTAL_ACTIONS = 33344, POSSIBLE_ACTIONS = 1500;
extern std::array<MoveAction, TOTAL_ACTIONS> action_list;
extern std::unordered_map<size_t, size_t> action_map;
void load_actions();
// 全局常量
const int N = 8;
const int EMPTY = 0;
const int BLACK = 1;
const int WHITE = -1;
const int BLOCK = 2;
const int dx[] = {-1, -1, -1, 0, 0, 1, 1, 1};
const int dy[] = {-1, 0, 1, -1, 1, -1, 0, 1};

// 全局变量声明
extern int thresold;
extern int start, current, turnID;
extern int currBotColor;
extern double C_UCT;

// Board 类声明
class Board
{
public:
    Board();
    bool isValid_Map(int x, int y) const;
    bool isBlank(int x, int y) const;
    bool isBlock(int x, int y) const;
    bool canDo(int x, int y, int nx, int ny) const;
    int getPiece(int x, int y);
    void clear(int x, int y);
    void restore(int x, int y, int nx, int ny, int bx, int by);
    void movePiece(int x, int y, int nx, int ny);
    void placeBlock(int x, int y);
    void print();
    void clear_all();

private:
    int temp;
    int board_[N][N];
};
extern Board board;
// Node 类声明
class Node
{
public:
    struct coordinates
    {
        int x, y;
    };
    struct Action
    {
        int startX, startY, endX, endY, barrierX, barrierY;
    };

    int player;
    int visits;
    int numberOfVisits;
    double value;
    bool cut;
    Node *parent;
    std::vector<Node *> children;
    coordinates myPiece[4];
    coordinates opponentPiece[4];
    std::vector<Action> legalMove;
    Action action;

    Node(int player);
};

// Evaluate 类声明
class Evaluate
{
public:
    Evaluate(Node *node);
    void init();
    uint64_t shiftMask(uint64_t a, int shift, uint64_t num, uint64_t direction, uint64_t can);
    uint64_t applyShifts(uint64_t temp, uint64_t mask, const int shiftDirections[], const uint64_t shiftAmounts[], const uint64_t directions[]);
    void queenBFS();
    void kingBFS();
    void pretreatment();
    std::pair<std::pair<double, double>, std::pair<double, double>> computeParameter();
    double computeBlankValue();
    double getEvaluateValues();

private:
    Node *node;
    int mobValues[8][8];
    int qcnt1 = 0, qcnt2 = 0, kcnt1 = 0, kcnt2 = 0, maxq = 0, maxk = 0;
    uint64_t umap = 0, p1 = 0, p2 = 0, blank = 0;
    uint64_t uq[40] = {0ull}, uk[40] = {0ull}, q1[40] = {0ull}, q2[40] = {0ull}, k1[40] = {0ull}, k2[40] = {0ull};
    double depthParameter[7] = {0.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125};
    double cntParameter[40] = {
        0.0,
        1.0,
        0.95,
        0.9,
        0.8,
        0.5,
        0.5,
    };
    double w1[56] = {0, 0, 0.074938141, 2.113308430, 1.829644322, 1.657095432, 1.844633341, 1.709531188, 1.947853684, 1.597983241, 1.290230632, 1.167080998, 0.580433190, 0.053694796, -0.011891359, 0.527658403, 0.975054741, 1.056464434, 0.872889936, 0.678786278, 0.323303133, 0.346939683, 0.260094881, 0.264616042, 0.246926725, 0.189162090, 0.141057417, 0.105003797, 0.100303024, 0.094548225, 0.075991668, 0.056678012, 0.052921258, 0.046266042, 0.047984783, 0.029863806, 0.046501521, 0.035372451, 0.036930695, 0.026028842, 0.023727236, 0.008815981, 0.005980607, 0.009275969, -0.003742765, -0.009044983, -0.010549444, -0.032312561, -0.012371800, -0.037411232, -0.032170087, -0.012665736, -0.025661280, 0.030489521, 0.038260937, -0.033962481};
    double w2[56] = {0, 0, -0.071123712, 1.850196242, 1.909757733, 1.786974669, 1.441033125, 1.831614137, 1.863318324, 1.549122095, 0.929738343, 0.985552371, 0.519130468, 0.652193844, 0.677611768, 0.715636075, -0.022837769, 0.100610048, -0.172568828, -0.087590173, 0.244123191, 0.132074401, 0.141869083, 0.097190522, 0.033922136, 0.084543742, 0.094968185, 0.085143305, 0.066933967, 0.070909552, 0.083291963, 0.085179411, 0.069383688, 0.058226194, 0.060640641, 0.065368392, 0.054329056, 0.054176800, 0.042966656, 0.044218585, 0.038709428, 0.031496748, 0.023774918, 0.015536518, 0.010674691, 0.005355561, 0.008374230, 0.010649841, 0.012298613, 0.012731116, 0.006807683, 0.003942049, 0.000137917, -0.014996326, -0.007956794, -0.035135169};
    double w3[56] = {0, 0, 0.285573512, 1.067316651, -0.211924806, 0.680028081, 0.517808735, 0.304819107, -0.360255659, 0.283211440, -0.118919544, -0.320547521, -0.164136052, -0.199558660, -0.158322647, -0.134399980, -0.166122779, -0.147537082, -0.119008608, -0.115313880, -0.088800281, -0.144199789, -0.093452334, -0.098844223, -0.061869688, -0.042582039, -0.026260521, -0.000397748, -0.004830216, -0.015821418, 0.015736405, 0.018657653, 0.027041025, 0.018133903, 0.027038572, 0.051898617, 0.024094300, 0.032939505, 0.027083304, 0.032979585, 0.024715593, 0.048389383, 0.051429220, 0.049864184, 0.065294459, 0.079495110, 0.070997894, 0.100243673, 0.052135076, 0.094104849, 0.087983906, 0.061890475, 0.079814047, -0.026632605, -0.043000557, 0.082739472};
    double w4[56] = {0, 0, 0.061070204, 2.073594332, 1.803056121, 1.969837666, 1.993263960, 1.986328840, 1.706978559, 1.858220458, 1.816977739, 1.320441842, 1.554476857, 1.503799796, 1.165330052, 0.306896836, 0.428119272, 0.127804548, 0.320249408, 0.250915736, 0.208399042, 0.209280223, 0.189255610, 0.176422641, 0.158465251, 0.123073861, 0.114771605, 0.107638910, 0.104587719, 0.094095841, 0.077729441, 0.085181594, 0.088369705, 0.096417032, 0.086386517, 0.084225371, 0.094143234, 0.101544440, 0.116312958, 0.118276447, 0.126188114, 0.131197393, 0.142377511, 0.138548672, 0.131359026, 0.133364707, 0.122829571, 0.130566165, 0.130473971, 0.133546650, 0.111887053, 0.097306497, 0.095673442, 0.109085456, 0.139188647, 0.08377216};
    double w5[56] = {0, 0, 0.107754663, -0.528144240, -1.379346251, -1.006276369, -0.825565398, -0.430609971, -0.473824382, -0.060512420, -0.942406714, -0.967272699, -0.992075503, -1.073423266, -0.858994186, -0.838604629, -0.738691866, -0.771107376, -0.667914927, -0.620147049, -0.567148566, -0.658371270, -0.578625798, -0.549303949, -0.488953888, -0.356694996, -0.272865117, -0.213500351, -0.175519645, -0.133494422, -0.070355214, -0.038640279, -0.043525659, -0.037141897, -0.031321511, -0.023366986, -0.021908367, -0.013590249, -0.019244174, -0.014974004, -0.009036653, -0.018890575, -0.004873865, -0.017690081, -0.012989196, -0.020389291, -0.031792857, -0.031873308, -0.028690575, -0.024877874, -0.033742540, -0.026430298, -0.027800240, -0.030750951, -0.021546466, -0.037860770};
    double w6[56] = {0, 0, -4.567995548, -3.716784000, -3.381752491, -2.938109636, -3.290686846, -2.702058077, -2.855453491, -2.595350504, -2.005551338, -1.637469649, -1.272953510, -1.047218084, -0.879562616, -0.746166408, -0.717715025, -0.592138231, -0.504874051, -0.366238832, -0.404548317, -0.305050164, -0.272765279, -0.249806121, -0.137687027, -0.254965365, -0.089192919, -0.171341106, -0.076427318, -0.220489189, -0.091823421, -0.180632398, -0.106192604, -0.145687878, -0.073370934, -0.149500415, -0.069663063, -0.189439490, -0.027017180, -0.123841494, -0.011969413, -0.017853998, 0.068564072, 0.002616666, 0.172109649, 0.157967970, 0.158981338, 0.351862490, 0.168605193, 0.400329232, 0.361347109, 0.220473915, 0.434309691, -0.030947627, -0.074839503, 0.44456172};
    const uint64_t West = 9187201950435737471ull, East = 18374403900871474942ull, South = 18446744073709551360ull, North = 72057594037927935ull;
    const uint64_t SE = South & East, SW = South & West, NE = North & East, NW = North & West;
};

// MCTS 类声明
class MCTS
{
public:
    MCTS(Node *node);
    void runSearch();
    std::pair<std::vector<int>, std::vector<double>> ans();

private:
    Node *select(Node *node);
    bool expandMove(Node &node);
    Node *expandNode(Node &node);
    void backpropagate(Node *node, double result);
    double computeUCT(const Node *node) const;
    struct coordinates
    {
        int x, y;
    };
    Node *root;
    const uint64_t border = 18446677844303282175ull, init = 2594215222373842980ull;
    uint64_t umoveB = 0, umoveE = 0;
};
class Game
{
public:
    std::pair<std::vector<int>, std::vector<double>> step(int turnID, py::array_t<int> action_idx);
};
#endif // GAME_H