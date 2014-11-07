package UFS;

import java.util.Arrays;

import weka.core.Instances;

public class EvaluatedUFS {

	
	
	public static double CA(Instances odata, int[] clusters)
	{
		double result = 0;
		double[] tmpdclass = odata.attributeToDoubleArray(odata.numAttributes()-1);
		int[] oclass = new int[odata.numInstances()];
		for(int i=0;i<tmpdclass.length;++i)
		{
			oclass[i]=(int)tmpdclass[i];
		}
		int[] tmpclass = oclass.clone();
		int[] tmpclusters = clusters.clone();
		
		Arrays.sort(tmpclusters);
		Arrays.sort(tmpclass);
		int[][] M = new int[tmpclass[tmpclass.length-1]+1][tmpclusters[tmpclusters.length-1]+1];
		
		for(int i=0;i<clusters.length;++i)
		{
			M[oclass[i]][clusters[i]]++;
		}
		for(int i=0;i<M.length;++i)
		{
			System.out.println(Arrays.toString(M[i]));
		}
		for(int i=0;i<M.length;++i)
		{
			int maxindex = -1;
			for(int j=0;j<M[0].length-1;++j)
			{
				if(M[i][j]<M[i][j+1])
					maxindex = j+1;
			}
			M[i][0]=maxindex;
		}

		for(int i=0;i<oclass.length;++i)
		{
			if(M[oclass[i]][0]==clusters[i])
				result++;
		}
		
		
		return (double)result/(double)odata.numInstances();
	}
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
