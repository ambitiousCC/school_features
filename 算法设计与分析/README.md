# 算法设计与分析

## 递归
* 时间复杂度： $O(2^k)$；实际上是生成树或遍历树的过程
* 定义：程序直接或间接的调用自己。
* 优点： 结构清晰，易读易证。
* 缺点：时间效率低，空间开销大，不容易优化。

### 二分查找

```c++
int bsearch(int b[], int x, int L, int R) 
{
	int mid;

	if(L>R) return -1;
	mid = (L+R) / 2;
	if(x == b[mid])
		return mid;
	else if(x<b[mid])
		return bsearch(b,x,L,mid-1);
	else
		return bsearch(b,x,mid+1,R);
}
```

## 阶乘

```c++
int fact(int n)
{
	if(n==1) return 1;
	else return n*fact(n-1);
}
```

### 斐波那契

```c++
long Fib(int n)
{
	if(n==0||n==1) return n;
	else return Fib(n-1) + Fib(n-2);
}
```

### 汉诺塔

```c++

/**
 *   | |   | |   | |
 *	 | |   | |   | |
 *    A     B     C
 */
void Hanoi(int n, char a, char b, char c) 
{
	if(n==1) cout<<"将第"<<n<<"个盘片从"<<a<<"移动到"<<c<<endl;
	else
	{
		Hanoi(n-1,a,c,b);// 把n-1个盘片从A借助C移动至B
		cout<<"将第"<<n<<"个盘片从"<<a<<"移动到"<<c<<endl;
		Hanoi(n-1,b,a,c);// 把n-1盘片从B借助A移动到C
	}
}
```

### 求顺序表中的最大值

```c++
int getMax(int L[], int i, int j)
{
	int mid;
	int max, max1, max2;
	if(i==j)
		max = L[i];
	else 
	{
		mid = (i+j) /2;
		max1 = getMax(L,i,mid);
		max2 = getMax(L,mid+1,j);
		max = (max1>max2)?max1:max2;
	}
	return max;
}
```

### 辗转相除法

```c++
int gcd(int m, int n)
{
	if(n==0) return m;
	else return gcd(n, m % n);
}
```

### 快速排序

```c++
void quickSort(int a[], int left, int right)
{
	int i,j,t,temp;
	if(left>right)
		return ; //不符合

	temp = a[left];
	i = left;
	j = right;

	while(i!=j)
	{
		while(a[j]>=temp && i<j)
			j--;
		while(a[i]<=temp && i<j)
			i++;

		if(i<j)
		{
			t=a[i];
			a[i]=a[j];
			a[j]=t;
		}
	}

	a[left] = a[i];
	a[i] = temp;

	quickSort(a,left,i-1);
	quickSort(a,i+1,right);
}
```

## 分治

将问题分解成为小的问题，然后小问题的解合并起来成为大问题的解。

1. 缩小规模可以解决问题
2. 具有最优子结构
3. 子问题解可以合并
4. 子问题相互独立

```c++
// logn
int bsearch(int b[], int x, int L, int R) 
{
	int left = 0;
	int right = n - 1;
	while(left<=right)
	{
		int mid = (left + right) /2;
		if(x==b[mid]) return mid;
		else if (x>b[mid]) left = mid + 1; //这里就和递归方法有所不同了
		else right = mid - 1;
	}
	return -1; //未找到
}
```

## 归并

```c++
void MergeSort(int a[], int left, int right)
{
	if(left<right)
	{
		int mid = (left+right) /2;
		MergeSort(a, left, mid); //对左边排序
		MergeSort(a, mid+1, right); //对右边界排序
		merge(a,b,left,mid+1,right); //合并元素到数组b
		copy(a,b,left,right); //复制回数组a
	}
}

//对于其中归并函数进行解读
void merge(int a[], int b[], int left, int right, int rightEnd)
{
	int leftEnd = right - 1;
	int temp = left;
	int n = rightEnd - left + 1;

	//归并开始
	while(left<=leftEnd && R<=rightEnd)
	{
		if(a[left]<=a[right])
			b[temp++] = a[left++];
		else
			b[temp++] = a[right++];
	}

	//多余部分
	while(left<=leftEnd)
		tempA[temp++] = a[left++];
	while(right<=rightEnd)
		tempA[temp++] = a[right++];;

	for(int i=0;i<n;i++,rightEnd--) {
		A[rightEnd] = temp[rightEnd];
	}
}
```

## 贪心策略

### 活动安排问题
早完成的活动先安排
```c++
void greatSelector(int n, int s[], int f[], bool a[])
{
	//注意：活动按照结束的时间的顺序排序
	a[1] = true; 
	int j = 1;
	for(int i=2;i<=n;i++)
	{
		if(s[i]>f[j])
		{
			a[i] = true;
			j = i;
		} else a[i] = false;
	}
}
```

### 背包问题

非0-1背包问题：0-1背包问题不能使用贪心算法
区分：0-1背包的物体不能够切分，而背包问题可以只装入物体的一部分

```c++
//思路是：计算各个物体的单位质量的价值：v/w，然后排序后装入
void Knapsack(int n, float M, float v[], float w[], float x[])
{
	//1. 计算各个物体的单位质量的价值：v/w，并排序
	Sort(n,v,w); // O(nlogn)

	int i;
	for(i=1; i<=n; i++) x[i] = 0; //初始化

	float c = M; //初始背包容量

	//2. 循环转入物体，如果装满了就退
	for(i=1; i<=n; i++)
	{
		if(w[i] > c) break;
		x[i] = 1;
		c -= w[i];
	}
	//3. 进一步判断是否有剩余
	if(i<=n) x[i] = c / w[i];
}
```

### 最优装载问题

x表示是否装入，w代表重量，c代表轮船当前的容量，n代表集装箱数目
```c++
void Loading(int x[], int w[], int c, int n)
{
	int *t = new int[n+1];

	//将集装箱按照其重量的从小到大的顺序排序，并不改变原来的w数组，而是将排序的结果存入t数组中
	Sort(w,t,n);

	//初始化
	for(int i=1; i<=n;i++) x[i] = 0;
	for(int i=1; i<=n&&w[t[i]]<=c;i++)
	{
		x[t[i]] = 1;
		c -= w[t[i]];
	}
}
```

### 多机调度问题

各个独立不可拆分作业，由机器加工处理，各个作业处理时间一直，每个作业都可以在一台机器上加工处理但是不可中断。

调度方案：所给的n个作业在尽可能短的时间由m机器加工完成

1. 排序作业，将作业按照时间从大到小排序
2. 选择完成时间最早的机器不断放入

注意：不一定得到最优解，因为此问题不具有贪心选择性质和最有结构


### 旅行商人问题

无向网，各个边的权值为路径长度
已知：连接矩阵w[][],连接的城市path[],p记录当前到达的城市，cost记录当前的路径长度，arrived[]记录城市是否到达

```c++
int tray_greedy(int n, int **w, int *path)
{
	//初始化
	for(int i=1;i<=n;i++) {
		arrived[i] = false;
		cost = 0;
	}

	//1. 从结点1出发
	path[1] = 1;
	p = 1;
	arrived[1] = true;

	//遍历路径
	int min;
	for(int i=2;i<=n;i++)
	{
		min = inf;//最小值初始化
		for(int j=1;j<=n;j++)
		{
			//3. 搜索到，如果没有去过，同时此时的路径最小
			if(!arrived[j]&&w[p][j]<min)
			{
				//选择为路径
				k=j;
				min = w[p][j];
			}
		}

		cost += w[p][k];
		path[i] = k;
		arrived[k] = true;
		p = k;
	}
	cost += w[p][1]; // 最后构成回路
	return cost;
}
```

### 最小生成树

1. Prim：从一个点出发，找最短的长度的边，找到后加入新的点，在新的点基础上继续寻找，每一次寻找不能形成环，最终所有点加入并构成树。O(n^2)两层循环

2. Kruskal：将权重从小到大排序，不断找最短的边，每一次寻找不能形成环，最终所有点加入并构成树。O(nlogn)

### 单源最短路径
dijistra算法具体过程，算法复杂度O(n^2)，外层循环n次，内存俩次循环。
![1](./dijistra.png)

### 霍夫曼编码
将新的或者旧的路径长度从小到大排序，并两两组合成新的值。最终构成一棵二叉树

算法复杂度为O(nlogn)

## 动态规划

基本要素：最优子结构和重叠子问题的性质

解决问题的关键：基本的递推关系式和恰当的边界条件

基本步骤
1. 找出最优的性质，并刻划其结构特性
2. 递归的定义最优值
3. 自底向上的方式计算出最优值
4. 根据计算最优值时得到的信息，构造最优解

### 数字三角形
核心：下一步的比较，下一步只能走D(r+1,j)或者D(r+1,j+1)

记住把每一步的结果可以保存起来，使用动态规划解决最优问题，必须最优解的每个局部解也是最优的