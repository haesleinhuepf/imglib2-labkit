package net.imglib2.atlas;

import net.imagej.ops.OpService;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.atlas.classification.Classifier;
import net.imglib2.cache.img.CellLoader;
import net.imglib2.cache.img.DiskCachedCellImgFactory;
import net.imglib2.cache.img.DiskCachedCellImgOptions;
import net.imglib2.img.Img;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.trainable_segmention.RevampUtils;
import net.imglib2.trainable_segmention.pixel_feature.calculator.FeatureCalculator;
import net.imglib2.trainable_segmention.pixel_feature.settings.FeatureSettings;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;

import java.util.List;
import java.util.stream.IntStream;

/**
 * @author Matthias Arzt
 */
// TODO : Refactor FeatureStack, what it is actually used for, can it be remove / replaced by something more appropriate
public class FeatureStack {

	private final MainFrame.Extensible extensible;

	private FeatureCalculator filter = null;

	private RandomAccessibleInterval<?> original;

	private List<RandomAccessibleInterval<FloatType>> slices;

	private CellGrid grid;

	private Notifier<Runnable> listeners = new Notifier<>();

	private RandomAccessibleInterval<?> preparedOriginal;

	public FeatureStack(MainFrame.Extensible extensible, RandomAccessibleInterval<?> original, Classifier classifier, boolean isTimeSeries) {
		this.extensible = extensible;
		this.original = original;
		this.grid = initGrid(original, isTimeSeries);
		classifier.listeners().add(c -> setFilter(c.settings()));
		setFilter(classifier.settings());
	}

	private static CellGrid initGrid(Interval interval, boolean isTimeSeries) {
		int[] cellDimension = initCellDimension(interval.numDimensions(), isTimeSeries);
		return new CellGrid(Intervals.dimensionsAsLongArray(interval), cellDimension);
	}

	private static int[] initCellDimension(int n, boolean isTimeSeries) {
		if(n == 3)
			return new int[] {256, 256, 4};
		return isTimeSeries ? RevampUtils.extend(initCellDimension(n - 1), 2) :
				initCellDimension(n);
	}

	private static int[] initCellDimension(int n) {
		int size = (int) Math.round(Math.pow(128. * 128., 1. / n) + 0.5);
		return IntStream.range(0, n).map(x -> size).toArray();
	}

	public void setFilter(FeatureSettings fs) {
		FeatureCalculator featureGroup = new FeatureCalculator(extensible.context().service(OpService.class), fs);
		if(filter != null && filter.equals(featureGroup))
			return;
		filter = featureGroup;
		preparedOriginal = prepareOriginal(original);
		RandomAccessible<?> extendedOriginal = Views.extendBorder(preparedOriginal);
		RandomAccessibleInterval<FloatType> block = cachedFeature(featureGroup, extendedOriginal);
		slices = RevampUtils.slices(block);
		listeners.forEach(Runnable::run);
	}

	private RandomAccessibleInterval<?> prepareOriginal(RandomAccessibleInterval<?> original) {
		Object voxel = original.randomAccess().get();
		if(voxel instanceof RealType && !(voxel instanceof FloatType))
			return AtlasUtils.toFloat(RevampUtils.uncheckedCast(original));
		return original;
	}

	private Img<FloatType> cachedFeature(FeatureCalculator feature, RandomAccessible<?> extendedOriginal) {
		return cachedFeatureBlock(feature, extendedOriginal, this.grid);
	}

	public static Img<FloatType> cachedFeatureBlock(FeatureCalculator feature, RandomAccessibleInterval<?> image) {
		return cachedFeatureBlock(feature, Views.extendBorder(image), initGrid(image, false));
	}

	public static Img<FloatType> cachedFeatureBlock(FeatureCalculator feature, RandomAccessible<?> extendedOriginal, CellGrid grid) {
		int count = feature.count();
		if(count <= 0)
			throw new IllegalArgumentException();
		long[] dimensions = AtlasUtils.extend(grid.getImgDimensions(), count);
		int[] cellDimensions = AtlasUtils.extend(new int[grid.numDimensions()], count);
		grid.cellDimensions(cellDimensions);
		final DiskCachedCellImgOptions featureOpts = DiskCachedCellImgOptions.options().cellDimensions( cellDimensions ).dirtyAccesses( false );
		final DiskCachedCellImgFactory< FloatType > featureFactory = new DiskCachedCellImgFactory<>( featureOpts );
		CellLoader<FloatType> loader = target -> feature.apply(extendedOriginal, RevampUtils.slices(target));
		return featureFactory.create(dimensions, new FloatType(), loader);
	}

	public List<RandomAccessibleInterval<FloatType>> slices() {
		return slices;
	}

	public FeatureCalculator filter() {
		return filter;
	}

	public Interval interval() {
		return new FinalInterval(original);
	}

	public CellGrid grid() {
		return grid;
	}

	public Notifier<Runnable> listeners() {
		return listeners;
	}

	public RandomAccessibleInterval<?> compatibleOriginal() {
		return preparedOriginal;
	}
}