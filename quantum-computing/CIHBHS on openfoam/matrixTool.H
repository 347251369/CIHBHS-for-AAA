#include <Python.h>
#include <iomanip>

struct matrix_tool
{
    // 按第一行展开计算|A|
    double getA(double arcs[150][150], int n)
    {
        if (n == 1)
        {
            return arcs[0][0];
        }
        double ans = 0;
        double temp[150][150] = {0.0};
        int i, j, k;
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n - 1; j++)
            {
                for (k = 0; k < n - 1; k++)
                {
                    temp[j][k] = arcs[j + 1][(k >= i) ? k + 1 : k];
                }
            }
            double t = getA(temp, n - 1);
            if (i % 2 == 0)
            {
                ans += arcs[0][i] * t;
            }
            else
            {
                ans -= arcs[0][i] * t;
            }
        }
        return ans;
    }

    // 计算每一行每一列的每个元素所对应的余子式，组成A*
    void getAStart(double arcs[150][150], int n, double ans[150][150])
    {
        if (n == 1)
        {
            ans[0][0] = 1;
            return;
        }
        int i, j, k, t;
        double temp[150][150];
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n; j++)
            {
                for (k = 0; k < n - 1; k++)
                {
                    for (t = 0; t < n - 1; t++)
                    {
                        temp[k][t] = arcs[k >= i ? k + 1 : k][t >= j ? t + 1 : t];
                    }
                }

                ans[j][i] = getA(temp, n - 1);
                if ((i + j) % 2 == 1)
                {
                    ans[j][i] = -ans[j][i];
                }
            }
        }
    }

    // 得到给定矩阵src的逆矩阵保存到des中。
    bool GetMatrixInverse(double src[150][150], int n, double des[150][150])
    {
        double flag = getA(src, n);
        double t[150][150];
        if (flag == 0)
        {
            return false;
        }
        else
        {
            getAStart(src, n, t);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    des[i][j] = t[i][j] / flag;
                }
            }
        }

        return true;
    }

    // 两矩阵相乘
    void matrixMutiplicate(double left[150][150],
                           double right[150][150],
                           double target[150][150],
                           int leftRowCount,
                           int rightColumnCount,
                           int elementCount)
    {
        for (int i = 0; i < leftRowCount; i++)
        {
            for (int j = 0; j < rightColumnCount; j++)
            {
                // 计算第i行j列第数据
                double result = 0;
                for (int p = 0; p < elementCount; p++)
                {
                    result = result + left[i][p] * right[p][j];
                }

                target[i][j] = result;
            }
        }
    }

    fvVectorMatrix transUEqn(fvVectorMatrix UEqn, volVectorField X)
    {
        // 获取矩阵的size
        int matrixSize = UEqn.diag().size();

        // 获取原始矩阵A
        double matrixA[150][150];
        for (int i = 0; i < matrixSize; i++)
            matrixA[i][i] = UEqn.diag()[i];
        for (int i = 0; i < UEqn.lduAddr().lowerAddr().size(); i++)
        {
            int l = UEqn.lduAddr().lowerAddr()[i];
            int u = UEqn.lduAddr().upperAddr()[i];
            matrixA[u][l] = UEqn.lower()[i];
            matrixA[l][u] = UEqn.upper()[i];
        }
        // 获取原始矩阵B
        double matrixB[150][150];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < matrixSize; j++)
                matrixB[j][i] = UEqn.source()[j][i];

        // 增加边界条件
        forAll(X.boundaryField(), patchI)
        {
            const fvPatch &pp = X.boundaryField()[patchI].patch();
            forAll(pp, faceI)
            {
                label cellI = pp.faceCells()[faceI];
                matrixA[cellI][cellI] = matrixA[cellI][cellI] + UEqn.internalCoeffs()[patchI][faceI][0];
                for (int i = 0; i < 3; i++)
                    matrixB[cellI][i] = matrixB[cellI][i] + UEqn.boundaryCoeffs()[patchI][faceI][i];
            }
        }

        // 转化矩阵C
        double matrixC[150][150];
        double a = matrixA[0][0];
        double b = matrixA[0][1];
        for (int i = 0; i < matrixSize; i++)
        {
            matrixC[i][i] = a;
            if (i != matrixSize - 1)
            {
                matrixC[i + 1][i] = b;
                matrixC[i][i + 1] = b;
            }
        }

        // 求解A的逆
        double inverseMatrixA[150][150];
        GetMatrixInverse(matrixA, matrixSize, inverseMatrixA);

        // 验证逆矩阵
        double testAResult[150][150];
        matrixMutiplicate(matrixA, inverseMatrixA, testAResult, matrixSize, matrixSize, matrixSize);

        // 求解D
        double matrixD[150][150];
        matrixMutiplicate(matrixC, inverseMatrixA, matrixD, matrixSize, matrixSize, matrixSize);

        // 求解b
        double newMatrixB[150][150];
        matrixMutiplicate(matrixD, matrixB, newMatrixB, matrixSize, 3, matrixSize);

        // 减去边界条件，并回填到UEqn
        forAll(X.boundaryField(), patchI)
        {
            const fvPatch &pp = X.boundaryField()[patchI].patch();
            forAll(pp, faceI)
            {
                label cellI = pp.faceCells()[faceI];
                matrixC[cellI][cellI] = matrixC[cellI][cellI] - UEqn.internalCoeffs()[patchI][faceI][0];
                for (int i = 0; i < 3; i++)
                    newMatrixB[cellI][i] = newMatrixB[cellI][i] - UEqn.boundaryCoeffs()[patchI][faceI][i];
            }
        }

        // 回填给UEqn
        for (int i = 0; i < matrixSize; i++)
            UEqn.diag()[i] = matrixC[i][i];
        for (int i = 0; i < UEqn.lduAddr().lowerAddr().size(); i++)
        {
            int l = UEqn.lduAddr().lowerAddr()[i];
            int u = UEqn.lduAddr().upperAddr()[i];
            UEqn.lower()[i] = matrixC[u][l];
            UEqn.upper()[i] = matrixC[l][u];
        }

        return UEqn;
    }
};
