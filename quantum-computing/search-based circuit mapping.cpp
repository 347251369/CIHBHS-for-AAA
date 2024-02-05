#include<iostream>
#include<fstream>
#include<algorithm>
#include<math.h>

using namespace std;

int n,m;
double point[250];
double hw[250][250];
int out[250];

int cir_n,cir_m;
double cirp[250];
struct edge
{
    int x;
    int y;
    int v;
    int op;
};
edge cirm[250];

int dis[250][250];
int path[250][250];
void floyd()
{
    for(int i=0;i<cir_n;i++)
    {
        for(int j=0;j<cir_n;j++)
        {
            int pointa=out[i];
            int pointb=out[j];
            path[pointa][pointb]=pointb;
            if(i!=j)
            {
                if(abs(hw[pointa][pointb]-100.0)>1e-4)dis[pointa][pointb]=1;
                else dis[pointa][pointb]=250;
            }
            else
                dis[pointa][pointb]=0;
        }
    }

    for(int k=0;k<cir_n;k++)
    {
        for(int i=0;i<cir_n;i++)
        {
            for(int j=0;j<cir_n;j++)
            {
                int pointa=out[i],pointb=out[j],pointc=out[k];
                if(dis[pointa][pointc]+dis[pointc][pointb]<dis[pointa][pointb])
                {
                    dis[pointa][pointb]=dis[pointa][pointc]+dis[pointc][pointb];
                    path[pointa][pointb]=path[pointa][pointc];
                }
            }
        }
    }
}

double min_cost;
int record[250];
int num[100];
int num_record[100];
double search_cost(int pa,int pb,int eid)
{
    double cost=1.00;
    int nxt=path[pa][pb];
    if(nxt!=pb)
    {
        cost=cost * pow(1.00-hw[pa][nxt]/100.0,6 * cirm[eid].v);
        cost= cost * search_cost(nxt,pb,eid);
    }
    else
    {
        if(cirm[eid].op==1)
            cost=cost * pow(1.00-hw[pa][nxt]/100.0,cirm[eid].v);
        else
            cost= cost * pow(1.00-hw[pa][nxt]/100.0,3*cirm[eid].v);
    }
    return cost;
}

void dfs(int p, int id)
{
    if(id == cir_n)
    {
        floyd();
        for(int i=0;i<cir_n;i++)num[i]=i;
        do{
            double sum_cost=1.00;
            for(int i=0;i<cir_n;i++)sum_cost = sum_cost * pow( (1.00 - point[out[i]]/100.00) ,cirp[num[i]]);
            for(int i=0;i<cir_m;i++)
            {
                int a=0;
                for(a=0;a<cir_n;a++)
                    if(num[a]==cirm[i].x)break;
                int b=0;
                for(b=0;b<cir_n;b++)
                    if(num[b]==cirm[i].y)break;
                sum_cost = sum_cost * search_cost(out[a],out[b],i);
            }
            if(sum_cost>min_cost)
            {
                min_cost=sum_cost;
                for(int i=0;i<cir_n;i++)
                {
                    record[i]=out[i];
                    num_record[i]=num[i];
                }
            }

        }while (next_permutation(num, num+cir_n));
        return;
    }

    for(int nxt=p+1;nxt<n;nxt++)
    {
        if( abs(hw[p][nxt]-100.0)>1e-4 )
        {
            int flag=0;
            for(int j=0;j<id;j++)
                if(nxt==out[id])flag=1;

            if(flag==0)
            {
                out[id]=p;
                dfs(nxt,id+1);
            }
        }
    }
}

int main()
{
    for(int i=0;i<100;i++)
        point[i]=100;
    for(int i=0;i<100;i++)
        for(int j=0;j<100;j++)
            hw[i][j]=100;

    ifstream in("hardware_data.txt");
    in>>n;
    in>>m;
    for(int i=0;i<n;i++)
        in>>point[i];
    for(int i=0;i<m;i++)
    {
        int pointA,pointB;
        double value;
        in>>pointA;
        in>>pointB;
        in>>value;
        hw[pointA][pointB]=hw[pointB][pointA]=value;
    }

    ifstream in2("cir_data.txt");
    in2>>cir_n;
    in2>>cir_m;
    for(int i=0;i<cir_n;i++)
        in2>>cirp[i];
    for(int i=0;i<cir_m;i++)
    {
        in2>>cirm[i].x;
        in2>>cirm[i].y;
        in2>>cirm[i].v;
        in2>>cirm[i].op;
    }
    min_cost=0.00;
    for(int i=0;i<n;i++)
        dfs(i,0);
    cout<<"The optimized circuit mapping:"<<endl;
    for(int i=0;i<cir_n-1;i++)
        cout<<record[i]<<" ";
    cout<<record[cir_n-1]<<endl;

    cout<<"The corresponding topology diagram is:"<<endl;
    for(int i=0;i<cir_n-1;i++)
        cout<<num_record[i]<<" ";
    cout<<num_record[cir_n-1]<<endl;
    return 0;
}
