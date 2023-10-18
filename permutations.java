import java.math.BigInteger;

public class permutations{
	final static int PERMUTATIONS = 0;
	final static int DERANGEMENTS = 1;
	
	final static int n = 630;
	
	static {
		init (n);
	}
	
	static BigInteger [] factorials;
	static BigInteger [] [] [] cache;
	
	static void init (int n) {
		factorials = new BigInteger [n + 1];
		cache = new BigInteger [2] [n + 1] [n + 1];
		factorials [0] = BigInteger.ONE;
		for (int i = 1;i <= n;i++)
			factorials [i] = factorials [i - 1].multiply (BigInteger.valueOf (i));
	}
	
	static BigInteger count (int which,int k,int n) {
		if (cache [which] [n] [k] == null)
			cache [which] [n] [k] = compute (which,k,n);
		return cache [which ][n] [k];
	}
	
	static BigInteger compute (int which,int k,int n) {
		if (k == 1 && which == DERANGEMENTS)
			return BigInteger.ZERO;
		if (k == 0 || n == 0)
			return k == 0 && n == 0 ? BigInteger.ONE : BigInteger.ZERO;
		
		BigInteger result = BigInteger.ZERO;
		BigInteger factor = BigInteger.valueOf (k);
		BigInteger power = BigInteger.valueOf (k);
		for (int j = 1,nkj = n - k;nkj >= 0;j++,nkj -= k,power = power.multiply (factor)) {
			BigInteger sum = BigInteger.ZERO;
			for (int t = 0;t < k && t <= nkj;t++)
				sum = sum.add (count (which,t,nkj));
			result = result.add (factorials [n].divide (factorials [j]).divide (factorials [nkj]).divide (power).multiply (sum));
		}
		return result;
	}
    public static void main(String[] args){
        System.out.println (count (PERMUTATIONS,100,200));
        // init(n);
    }

}