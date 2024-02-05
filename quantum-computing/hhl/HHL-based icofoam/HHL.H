#include <Python.h>
#include <iomanip>

struct Harrow_Hassidim_Lloyd
{
	PyObject *pModule;
	PyObject *pFunc;
	int n;
	int precision;
	double A[150][150];
	double B[3][150];
	double HA[40000];
    double HB[150];
    double result[150];
    int times;

	std::string array_to_string(int n, double *a)
	{
	    std::string sa;
	    sa.clear();
      std::stringstream ss;
      ss.clear();
	    for(int i=0;i<n-1;i++)
      {
	        ss<<std::setprecision(10)<< a[i];
	        sa = sa + ss.str() + ' ';
	        ss.clear();
	        ss.str("");
      }
      ss<<std::setprecision(10)<< a[n-1];
	    sa = sa + ss.str();
	    return sa;
	}

	void string_to_array(char *s, double* array)
	{
	    std::stringstream ss;
      ss.clear();
      ss.str("");
	    ss<<s;
	    int i=0;
	    while(!ss.eof() && i<40000)
	    {
	        ss>>array[i];
	        i++;
	    }
	}
 
	void change_precision(int p)
	{
		precision=p;
	}
	void init(int p)
	{
		Py_Initialize();
	    PyRun_SimpleString("import sys");
	    PyRun_SimpleString("sys.path.append('/home/openfoam2112/applications/OpenFOAM-v2112/applications/solvers/incompressible/Myicofoam/')");
	    PyRun_SimpleString("sys.path.append('/home/openfoam2112/.local/lib/python3.8/site-packages')");
	    PyRun_SimpleString("print(sys.version)");

	    pModule = PyImport_ImportModule("calculator");
	    pFunc = PyObject_GetAttrString(pModule, "cal");
	    precision = p;
     times =1;
	}
 
	void get_AB(fvVectorMatrix UEqn, volVectorField X)
	{
		n=UEqn.diag().size();
		//A matrix
		for(int i=0;i<n;i++)A[i][i]=UEqn.diag()[i];
        for(int i=0;i<UEqn.lduAddr().lowerAddr().size();i++)
        {
            int l=UEqn.lduAddr().lowerAddr()[i];
            int u=UEqn.lduAddr().upperAddr()[i];
            A[u][l]=UEqn.lower()[i];
            A[l][u]=UEqn.upper()[i];
        }
        //B matrix
        for(int i=0;i<3;i++)
            for(int j=0;j<n;j++)
               B[i][j]=UEqn.source()[j][i];
        //ADD boundary to A and B matrix
        forAll(X.boundaryField(), patchI)
        {
            const fvPatch &pp = X.boundaryField()[patchI].patch();
            forAll(pp, faceI) 
            {
                label cellI = pp.faceCells()[faceI];
                A[cellI][cellI] = A[cellI][cellI] + UEqn.internalCoeffs()[patchI][faceI][0];
                for (int i=0;i<3;i++)
                    B[i][cellI] = B[i][cellI] + UEqn.boundaryCoeffs()[patchI][faceI][i];
            }
        }
	}
 
 
	void solve_by_virtual_machine(double *HA, double *HB, double *result)
	{
	    std::string sa = array_to_string(n*n, HA);
	    std::string sb = array_to_string(n, HB);
        Info<<(times++)<<" ::"<<endl;
	    PyObject *pRet = PyEval_CallObject(pFunc, Py_BuildValue("ssi",sa.c_str(), sb.c_str(), precision));
	    char *res_s;
	    PyArg_Parse(pRet, "s", &res_s);
	    string_to_array(res_s, result);
	}
	void vir_run(volVectorField& X)
	{
        for(int i=0;i<n;i++)
          for(int j=0;j<n;j++)
            HA[i*n+j]=A[i][j]; 
        
        for(int i=0;i<3;i++)
        {
          for(int j=0;j<n;j++)HB[j]=B[i][j];
          
          
          solve_by_virtual_machine(HA,HB,result);
          for(int j=0;j<n;j++)X[j][i]=result[j];
        }
	}
};