#pragma once
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>
constexpr const int N = 8;
struct move
{
    uint8_t start, go, arrow;
};
struct DisArr
{
    uint64_t a[38];
    int len;
    void clear() { memset(a, 0, (len + 1) * sizeof(uint64_t)); }
};
namespace dir
{
    static constexpr uint64_t W = 9187201950435737471ull, E = 18374403900871474942ull,
                              S = 18446744073709551360ull, N = 72057594037927935ull,
                              SE = S & E, SW = S & W, NE = N & E, NW = N & W;
}
struct board
{
    uint64_t a, b, c;
    board(uint64_t A, uint64_t B, uint64_t C) : a(A), b(B), c(C) {}
    board Move(int p0, int p1, int p2) const;
    board Move(const move &x) const;
    double eval() const;
    board reverse() const;
};
namespace tables
{
    extern double arr[][6];
}
inline void up(double &x, double y)
{
    if (x < y)
        x = y;
}
inline bool good(int x, int y) { return (uint32_t)x < N && (uint32_t)y < N; }

inline void flip(uint64_t &s, int x, int y) { s ^= (uint64_t)1 << (x * 8 + y); }
inline int id(int x, int y) { return x * N | y; }
inline int row(uint32_t x) { return x / 8; }
inline int col(uint32_t x) { return x % 8; }
double eval(const board &x);
class Evaluate
{
public:
    Evaluate(uint64_t u1, uint64_t u2, uint64_t u3) : b(u1, u2, u3) {}
    double eval() { return b.eval(); }

private:
    board b;
};