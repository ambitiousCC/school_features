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

