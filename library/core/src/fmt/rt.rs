#![allow(missing_debug_implementations)]
#![unstable(feature = "fmt_internals", reason = "internal to format_args!", issue = "none")]

//! These are the lang items used by format_args!().

use crate::fmt::{
    self, Binary, Debug, Display, Formatter, LowerExp, LowerHex, Octal, Pointer, UpperExp, UpperHex,
};
#[cfg(not(bootstrap))]
use crate::fmt::{FormattingOptions, Write};
use crate::marker::PhantomData;
use crate::ptr::NonNull;
#[cfg(not(bootstrap))]
use crate::str;
#[cfg(bootstrap)]
use crate::{hint, mem};

#[cfg(bootstrap)]
#[lang = "format_placeholder"]
#[derive(Copy, Clone)]
pub struct Placeholder {
    pub position: usize,
    pub fill: char,
    pub align: Alignment,
    pub flags: u32,
    pub precision: Count,
    pub width: Count,
}

#[cfg(bootstrap)]
impl Placeholder {
    #[inline]
    pub const fn new(
        position: usize,
        fill: char,
        align: Alignment,
        flags: u32,
        precision: Count,
        width: Count,
    ) -> Self {
        Self { position, fill, align, flags, precision, width }
    }
}

#[cfg(bootstrap)]
#[lang = "format_alignment"]
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Alignment {
    Left,
    Right,
    Center,
    Unknown,
}

/// Used by [width](https://doc.rust-lang.org/std/fmt/#width)
/// and [precision](https://doc.rust-lang.org/std/fmt/#precision) specifiers.
#[cfg(bootstrap)]
#[lang = "format_count"]
#[derive(Copy, Clone)]
pub enum Count {
    /// Specified with a literal number, stores the value
    Is(usize),
    /// Specified using `$` and `*` syntaxes, stores the index into `args`
    Param(usize),
    /// Not specified
    Implied,
}

// This needs to match the order of flags in compiler/rustc_ast_lowering/src/format.rs.
#[derive(Copy, Clone)]
pub(super) enum Flag {
    SignPlus,
    SignMinus,
    Alternate,
    SignAwareZeroPad,
    DebugLowerHex,
    DebugUpperHex,
}

#[cfg(bootstrap)]
#[derive(Copy, Clone)]
enum ArgumentType<'a> {
    Placeholder {
        // INVARIANT: `formatter` has type `fn(&T, _) -> _` for some `T`, and `value`
        // was derived from a `&'a T`.
        value: NonNull<()>,
        formatter: unsafe fn(NonNull<()>, &mut Formatter<'_>) -> fmt::Result,
        _lifetime: PhantomData<&'a ()>,
    },
    Count(usize),
}

/// This struct represents a generic "argument" which is taken by format_args!().
///
/// This can be either a placeholder argument or a count argument.
/// * A placeholder argument contains a function to format the given value. At
///   compile time it is ensured that the function and the value have the correct
///   types, and then this struct is used to canonicalize arguments to one type.
///   Placeholder arguments are essentially an optimized partially applied formatting
///   function, equivalent to `exists T.(&T, fn(&T, &mut Formatter<'_>) -> Result`.
/// * A count argument contains a count for dynamic formatting parameters like
///   precision and width.
#[cfg(bootstrap)]
#[lang = "format_argument"]
#[derive(Copy, Clone)]
pub struct Argument<'a> {
    ty: ArgumentType<'a>,
}

#[cfg(bootstrap)]
#[rustc_diagnostic_item = "ArgumentMethods"]
impl Argument<'_> {
    #[inline]
    const fn new<'a, T>(x: &'a T, f: fn(&T, &mut Formatter<'_>) -> fmt::Result) -> Argument<'a> {
        Argument {
            // INVARIANT: this creates an `ArgumentType<'a>` from a `&'a T` and
            // a `fn(&T, ...)`, so the invariant is maintained.
            ty: ArgumentType::Placeholder {
                value: NonNull::from_ref(x).cast(),
                // SAFETY: function pointers always have the same layout.
                formatter: unsafe { mem::transmute(f) },
                _lifetime: PhantomData,
            },
        }
    }

    #[inline]
    pub fn new_display<T: Display>(x: &T) -> Argument<'_> {
        Self::new(x, Display::fmt)
    }
    #[inline]
    pub fn new_debug<T: Debug>(x: &T) -> Argument<'_> {
        Self::new(x, Debug::fmt)
    }
    #[inline]
    pub fn new_debug_noop<T: Debug>(x: &T) -> Argument<'_> {
        Self::new(x, |_, _| Ok(()))
    }
    #[inline]
    pub fn new_octal<T: Octal>(x: &T) -> Argument<'_> {
        Self::new(x, Octal::fmt)
    }
    #[inline]
    pub fn new_lower_hex<T: LowerHex>(x: &T) -> Argument<'_> {
        Self::new(x, LowerHex::fmt)
    }
    #[inline]
    pub fn new_upper_hex<T: UpperHex>(x: &T) -> Argument<'_> {
        Self::new(x, UpperHex::fmt)
    }
    #[inline]
    pub fn new_pointer<T: Pointer>(x: &T) -> Argument<'_> {
        Self::new(x, Pointer::fmt)
    }
    #[inline]
    pub fn new_binary<T: Binary>(x: &T) -> Argument<'_> {
        Self::new(x, Binary::fmt)
    }
    #[inline]
    pub fn new_lower_exp<T: LowerExp>(x: &T) -> Argument<'_> {
        Self::new(x, LowerExp::fmt)
    }
    #[inline]
    pub fn new_upper_exp<T: UpperExp>(x: &T) -> Argument<'_> {
        Self::new(x, UpperExp::fmt)
    }
    #[inline]
    pub const fn from_usize(x: &usize) -> Argument<'_> {
        Argument { ty: ArgumentType::Count(*x) }
    }

    /// Format this placeholder argument.
    ///
    /// # Safety
    ///
    /// This argument must actually be a placeholder argument.
    ///
    // FIXME: Transmuting formatter in new and indirectly branching to/calling
    // it here is an explicit CFI violation.
    #[allow(inline_no_sanitize)]
    #[no_sanitize(cfi, kcfi)]
    #[inline]
    pub(super) unsafe fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.ty {
            // SAFETY:
            // Because of the invariant that if `formatter` had the type
            // `fn(&T, _) -> _` then `value` has type `&'b T` where `'b` is
            // the lifetime of the `ArgumentType`, and because references
            // and `NonNull` are ABI-compatible, this is completely equivalent
            // to calling the original function passed to `new` with the
            // original reference, which is sound.
            ArgumentType::Placeholder { formatter, value, .. } => unsafe { formatter(value, f) },
            // SAFETY: the caller promised this.
            ArgumentType::Count(_) => unsafe { hint::unreachable_unchecked() },
        }
    }

    #[inline]
    pub(super) const fn as_usize(&self) -> Option<usize> {
        match self.ty {
            ArgumentType::Count(count) => Some(count),
            ArgumentType::Placeholder { .. } => None,
        }
    }

    /// Used by `format_args` when all arguments are gone after inlining,
    /// when using `&[]` would incorrectly allow for a bigger lifetime.
    ///
    /// This fails without format argument inlining, and that shouldn't be different
    /// when the argument is inlined:
    ///
    /// ```compile_fail,E0716
    /// let f = format_args!("{}", "a");
    /// println!("{f}");
    /// ```
    #[inline]
    pub const fn none() -> [Self; 0] {
        []
    }
}

#[cfg(not(bootstrap))]
type FmtFn = unsafe fn(state: &mut State<'_>) -> fmt::Result;

#[cfg(not(bootstrap))]
union ArgumentInner {
    fmt_fn: FmtFn,

    // Points to one of:
    //
    // - `[&'static str]`: All literal pieces used in `format_args`.
    // - An argument used in placeholders.
    ptr: NonNull<()>,

    // Estimated capacity, width or precision value specified at runtime.
    usize: usize,
}

#[cfg(not(bootstrap))]
#[lang = "format_argument"]
pub struct Argument<'a> {
    inner: ArgumentInner,
    _phantom_data: PhantomData<&'a ()>,
}

#[cfg(not(bootstrap))]
#[rustc_diagnostic_item = "ArgumentMethods"]
impl<'a> Argument<'a> {
    #[inline(always)]
    pub const fn from_fmt_fn(fmt_fn: FmtFn) -> Self {
        Self { inner: ArgumentInner { fmt_fn }, _phantom_data: PhantomData }
    }

    #[inline(always)]
    pub const fn from_ref<T>(value: &'a T) -> Self {
        Self {
            inner: ArgumentInner { ptr: NonNull::from_ref(value).cast() },
            _phantom_data: PhantomData,
        }
    }

    #[inline(always)]
    pub const fn from_compile_time_data<const N: usize>(
        compile_time_data: &'static CompileTimeData<N>,
    ) -> Self {
        Self::from_ref(compile_time_data)
    }

    #[inline(always)]
    pub fn from_usize(value: &usize) -> Self {
        Self { inner: ArgumentInner { usize: *value }, _phantom_data: PhantomData }
    }

    /// # Safety
    ///
    /// `Argument` contains a `FmtFn` object.
    #[inline(always)]
    pub unsafe fn as_fmt_fn(&self) -> FmtFn {
        // SAFETY: Guaranteed by caller.
        unsafe { self.inner.fmt_fn }
    }

    /// # Safety
    ///
    /// `Argument` contains a reference object that points to a `T`.
    #[inline(always)]
    pub const unsafe fn as_ptr<T>(&self) -> NonNull<T> {
        // SAFETY: Guaranteed by caller.
        unsafe { self.inner.ptr.cast() }
    }

    /// # Safety
    ///
    /// `Argument` contains a reference object that points to a `T`.
    #[inline(always)]
    pub const unsafe fn as_ref<T>(&self) -> &'a T {
        // SAFETY: Guaranteed by caller.
        unsafe { self.as_ptr().as_ref() }
    }

    /// # Safety
    ///
    /// This object is created with `from_usize`.
    #[inline(always)]
    pub unsafe fn as_usize(&self) -> usize {
        // SAFETY: Guaranteed by caller.
        unsafe { self.inner.usize }
    }

    #[inline(always)]
    pub const fn noop() {}
}

/// Mutable data when running a formatting process.
#[cfg(not(bootstrap))]
pub struct State<'a> {
    formatter: Formatter<'a>,
    literal_ptr: NonNull<&'static str>,
    argument_ptr: NonNull<Argument<'a>>,
}

#[cfg(not(bootstrap))]
impl<'a> State<'a> {
    pub fn new(
        output: &'a mut (dyn Write + 'a),
        literal_ptr: NonNull<&'static str>,
        argument_ptr: NonNull<Argument<'a>>,
    ) -> Self {
        Self {
            formatter: Formatter::new(output, const { FormattingOptions::new() }),
            literal_ptr,
            argument_ptr,
        }
    }

    /// # Safety
    ///
    /// Offsetting `literal_ptr` should not go out of bounds of its allocation.
    unsafe fn offset_literal_ptr<const N: isize>(&mut self) {
        // SAFETY: Guaranteed by caller.
        self.literal_ptr = unsafe { self.literal_ptr.offset(N) };
    }

    /// # Safety
    ///
    /// Offsetting `argument_ptr` should not go out of bounds of its allocation.
    unsafe fn offset_argument_ptr<const N: isize>(&mut self) {
        // SAFETY: Guaranteed by caller.
        self.argument_ptr = unsafe { self.argument_ptr.offset(N) };
    }

    /// # Safety
    ///
    /// `self.literal_ptr` is currently pointed to a valid `&'static str` object.
    unsafe fn write_str(&mut self) -> fmt::Result {
        // SAFETY: Guaranteed by caller.
        self.formatter.write_str(*unsafe { self.literal_ptr.as_ref() })
    }

    fn set_flags<const FLAGS: u32>(&mut self) {
        self.formatter.options.flags = FLAGS;
    }

    fn set_fill<const FILL: char>(&mut self) {
        self.formatter.options.fill = FILL;
    }

    fn clear_align(&mut self) {
        self.formatter.options.align = None;
    }

    fn set_left_align(&mut self) {
        self.formatter.options.align = Some(fmt::Alignment::Left);
    }

    fn set_right_align(&mut self) {
        self.formatter.options.align = Some(fmt::Alignment::Right);
    }

    fn set_center_align(&mut self) {
        self.formatter.options.align = Some(fmt::Alignment::Center);
    }

    fn clear_width(&mut self) {
        self.formatter.options.width = None;
    }

    fn set_width<const W: usize>(&mut self) {
        self.formatter.options.width = Some(W);
    }

    /// # Safety
    ///
    /// `self.argument_ptr` is currently pointed to a valid `Argument` object that is created with
    /// `Argument::from_usize`.
    unsafe fn set_dyn_width(&mut self) {
        // SAFETY: Guaranteed by caller.
        self.formatter.options.width = Some(unsafe { self.argument_ptr.as_ref().as_usize() });
    }

    fn clear_precision(&mut self) {
        self.formatter.options.precision = None;
    }

    fn set_precision<const P: usize>(&mut self) {
        self.formatter.options.precision = Some(P);
    }

    /// # Safety
    ///
    /// `self.argument_ptr` is currently pointed to a valid `Argument` object that is created with
    /// `Argument::from_usize`.
    unsafe fn set_dyn_precision(&mut self) {
        // SAFETY: Guaranteed by caller.
        self.formatter.options.precision = Some(unsafe { self.argument_ptr.as_ref().as_usize() });
    }

    /// # Safety
    ///
    /// `self.argument_ptr` is currently pointed to a valid `Argument` object that is created with
    /// `Argument::ref_builder::<T>`.
    #[inline(never)]
    unsafe fn fmt_binary<T>(&mut self) -> fmt::Result
    where
        T: Binary,
    {
        // SAFETY: Guaranteed by caller.
        Binary::fmt(unsafe { self.argument_ptr.as_ref().as_ref::<T>() }, &mut self.formatter)
    }

    /// # Safety
    ///
    /// `self.argument_ptr` is currently pointed to a valid `Argument` object that is created with
    /// `Argument::ref_builder::<T>`.
    #[inline(never)]
    unsafe fn fmt_debug<T>(&mut self) -> fmt::Result
    where
        T: Debug,
    {
        // SAFETY: Guaranteed by caller.
        Debug::fmt(unsafe { self.argument_ptr.as_ref().as_ref::<T>() }, &mut self.formatter)
    }

    /// # Safety
    ///
    /// `self.argument_ptr` is currently pointed to a valid `Argument` object that is created with
    /// `Argument::ref_builder::<T>`.
    #[inline(never)]
    unsafe fn fmt_display<T>(&mut self) -> fmt::Result
    where
        T: Display,
    {
        // SAFETY: Guaranteed by caller.
        Display::fmt(unsafe { self.argument_ptr.as_ref().as_ref::<T>() }, &mut self.formatter)
    }

    /// # Safety
    ///
    /// `self.argument_ptr` is currently pointed to a valid `Argument` object that is created with
    /// `Argument::ref_builder::<T>`.
    #[inline(never)]
    unsafe fn fmt_lower_exp<T>(&mut self) -> fmt::Result
    where
        T: LowerExp,
    {
        // SAFETY: Guaranteed by caller.
        LowerExp::fmt(unsafe { self.argument_ptr.as_ref().as_ref::<T>() }, &mut self.formatter)
    }

    /// # Safety
    ///
    /// `self.argument_ptr` is currently pointed to a valid `Argument` object that is created with
    /// `Argument::ref_builder::<T>`.
    #[inline(never)]
    unsafe fn fmt_lower_hex<T>(&mut self) -> fmt::Result
    where
        T: LowerHex,
    {
        // SAFETY: Guaranteed by caller.
        LowerHex::fmt(unsafe { self.argument_ptr.as_ref().as_ref::<T>() }, &mut self.formatter)
    }

    /// # Safety
    ///
    /// `self.argument_ptr` is currently pointed to a valid `Argument` object that is created with
    /// `Argument::ref_builder::<T>`.
    #[inline(never)]
    unsafe fn fmt_octal<T>(&mut self) -> fmt::Result
    where
        T: Octal,
    {
        // SAFETY: Guaranteed by caller.
        Octal::fmt(unsafe { self.argument_ptr.as_ref().as_ref::<T>() }, &mut self.formatter)
    }

    /// # Safety
    ///
    /// `self.argument_ptr` is currently pointed to a valid `Argument` object that is created with
    /// `Argument::ref_builder::<T>`.
    #[inline(never)]
    unsafe fn fmt_pointer<T>(&mut self) -> fmt::Result
    where
        T: Pointer,
    {
        // SAFETY: Guaranteed by caller.
        Pointer::fmt(unsafe { self.argument_ptr.as_ref().as_ref::<T>() }, &mut self.formatter)
    }

    /// # Safety
    ///
    /// `self.argument_ptr` is currently pointed to a valid `Argument` object that is created with
    /// `Argument::ref_builder::<T>`.
    #[inline(never)]
    unsafe fn fmt_upper_exp<T>(&mut self) -> fmt::Result
    where
        T: UpperExp,
    {
        // SAFETY: Guaranteed by caller.
        UpperExp::fmt(unsafe { self.argument_ptr.as_ref().as_ref::<T>() }, &mut self.formatter)
    }

    /// # Safety
    ///
    /// `self.argument_ptr` is currently pointed to a valid `Argument` object that is created with
    /// `Argument::ref_builder::<T>`.
    #[inline(never)]
    unsafe fn fmt_upper_hex<T>(&mut self) -> fmt::Result
    where
        T: UpperHex,
    {
        // SAFETY: Guaranteed by caller.
        UpperHex::fmt(unsafe { self.argument_ptr.as_ref().as_ref::<T>() }, &mut self.formatter)
    }
}

/// A helper trait that defines a format operation sequence.
#[cfg(not(bootstrap))]
pub trait FmtOp {
    unsafe fn fmt(state: &mut State<'_>) -> fmt::Result;
}

/// Like `PhantomData`, but is not intended for instantiating.
#[cfg(not(bootstrap))]
struct NeverPhantomData<T>
where
    T: ?Sized,
{
    _never: !,
    _phantom_data: PhantomData<T>,
}

/// Marks the end of a format operation sequence.
#[cfg(not(bootstrap))]
pub struct End(NeverPhantomData<()>);

#[cfg(not(bootstrap))]
pub struct OffsetLiteralPtr<const N: isize, F>(NeverPhantomData<F>);

#[cfg(not(bootstrap))]
impl<const N: isize, F> FmtOp for OffsetLiteralPtr<N, F>
where
    F: FmtOp,
{
    /// # Safety
    ///
    /// After offsetting literal pointer by `N`, `state` should satisfy the safety requirement of
    /// `F::fmt`.
    unsafe fn fmt(state: &mut State<'_>) -> fmt::Result {
        // SAFETY: Guaranteed by caller.
        unsafe {
            state.offset_literal_ptr::<N>();

            F::fmt(state)
        }
    }
}

#[cfg(not(bootstrap))]
pub struct OffsetArgumentPtr<const N: isize, F>(NeverPhantomData<F>);

#[cfg(not(bootstrap))]
impl<const N: isize, F> FmtOp for OffsetArgumentPtr<N, F>
where
    F: FmtOp,
{
    /// # Safety
    ///
    /// After offsetting argument pointer by `N`, `state` should satisfy the safety requirement of
    /// `F::fmt`.
    unsafe fn fmt(state: &mut State<'_>) -> fmt::Result {
        // SAFETY: Guaranteed by caller.
        unsafe {
            state.offset_argument_ptr::<N>();

            F::fmt(state)
        }
    }
}

#[cfg(not(bootstrap))]
pub struct WriteStr<F>(NeverPhantomData<F>);

#[cfg(not(bootstrap))]
impl<F> FmtOp for WriteStr<F>
where
    F: FmtOp,
{
    /// # Safety
    ///
    /// - Literal pointer in `state` should points to an `&'static str` object.
    /// - `state` should satisfy the safety requirement of `F::fmt`.
    unsafe fn fmt(state: &mut State<'_>) -> fmt::Result {
        // SAFETY: Guaranteed by caller.
        unsafe {
            state.write_str()?;

            F::fmt(state)
        }
    }
}

#[cfg(not(bootstrap))]
impl FmtOp for WriteStr<End> {
    /// # Safety
    ///
    /// Literal pointer in `state` should points to an `&'static str` object.
    unsafe fn fmt(state: &mut State<'_>) -> fmt::Result {
        // SAFETY: Guaranteed by caller.
        unsafe { state.write_str() }
    }
}

#[cfg(not(bootstrap))]
macro_rules! define_formatter_options_ops {
    (
        $(
            #[fmt_safety_doc = $fmt_safety_doc:literal]
            $name:ident $(<const $const_generic:ident: $const_generic_type:ty>)?
            =>
            $method:ident,
        )*
    ) => {
        $(
            pub struct $name<$(const $const_generic: $const_generic_type,)? F>(NeverPhantomData<F>);

            impl <$(const $const_generic: $const_generic_type,)? F> FmtOp for $name<$($const_generic,)* F>
            where
                F: FmtOp,
            {
                /// # Safety
                ///
                #[doc = $fmt_safety_doc]
                /// - After calling the method above, it is safe to call `F::fmt(state)`.
                unsafe fn fmt(state: &mut State<'_>) -> fmt::Result {
                    // SAFETY: Guaranteed by caller.
                    unsafe {
                        state.$method$(::<$const_generic>)?();

                        F::fmt(state)
                    }
                }
            }
        )*
    };
}

#[cfg(not(bootstrap))]
define_formatter_options_ops![
    #[fmt_safety_doc = "It is safe to call `state.set_flags::<FLAGS>()`."]
    SetFlags<const FLAGS: u32>   => set_flags,

    #[fmt_safety_doc = "It is safe to call `state.set_fill::<FILL>()`."]
    SetFill<const FILL: char>    => set_fill,

    #[fmt_safety_doc = "It is safe to call `state.set_clear_align()`."]
    ClearAlign                   => clear_align,

    #[fmt_safety_doc = "It is safe to call `state.set_left_align()`."]
    SetLeftAlign                 => set_left_align,

    #[fmt_safety_doc = "It is safe to call `state.set_right_align()`."]
    SetRightAlign                => set_right_align,

    #[fmt_safety_doc = "It is safe to call `state.set_center_align()`."]
    SetCenterAlign               => set_center_align,

    #[fmt_safety_doc = "It is safe to call `state.clear_with()`."]
    ClearWidth                   => clear_width,

    #[fmt_safety_doc = "It is safe to call `state.set_width::<W>()`."]
    SetWidth<const W: usize>     => set_width,

    #[fmt_safety_doc = "It is safe to call `state.set_dyn_width()`."]
    SetDynWidth                  => set_dyn_width,

    #[fmt_safety_doc = "It is safe to call `state.clear_precision()`."]
    ClearPrecision               => clear_precision,

    #[fmt_safety_doc = "It is safe to call `state.set_precision()`."]
    SetPrecision<const P: usize> => set_precision,

    #[fmt_safety_doc = "It is safe to call `state.set_dyn_width()`."]
    SetDynPrecision              => set_dyn_precision,
];

#[cfg(not(bootstrap))]
macro_rules! define_fmt_trait_ops {
    ($($name:ident => $method:ident, $trait:ident,)*) => {
        $(
            pub struct $name<T, F>(NeverPhantomData<T>, NeverPhantomData<F>)
            where
                T: ?Sized;

            impl<T, F> FmtOp for $name<T, F>
            where
                T: $trait,
                F: FmtOp,
            {
                /// # Safety
                ///
                #[doc = concat!("- It is safe to call `state.", stringify!($method), "::<T>()`.")]
                /// - After calling the method above, it is safe to call `F::fmt(state)`.
                unsafe fn fmt(state: &mut State<'_>) -> fmt::Result {
                    // SAFETY: Guaranteed by caller.
                    unsafe {
                        state.$method::<T>()?;

                        F::fmt(state)
                    }
                }
            }

            impl<T> FmtOp for $name<T, End>
            where
                T: $trait,
            {
                /// # Safety
                ///
                #[doc = concat!("- It is safe to call `state.", stringify!($method), "::<T>()`.")]
                unsafe fn fmt(state: &mut State<'_>) -> fmt::Result {
                    // SAFETY: Guaranteed by caller.
                    unsafe { state.$method::<T>() }
                }
            }
        )*
    };
}

#[cfg(not(bootstrap))]
define_fmt_trait_ops![
    FmtBinary   => fmt_binary,    Binary,
    FmtDebug    => fmt_debug,     Debug,
    FmtDisplay  => fmt_display,   Display,
    FmtLowerExp => fmt_lower_exp, LowerExp,
    FmtLowerHex => fmt_lower_hex, LowerHex,
    FmtOctal    => fmt_octal,     Octal,
    FmtPointer  => fmt_pointer,   Pointer,
    FmtUpperExp => fmt_upper_exp, UpperExp,
    FmtUpperHex => fmt_upper_hex, UpperHex,
];

#[cfg(not(bootstrap))]
#[lang = "format_fmt_fn_builder"]
pub struct FmtFnBuilder<F>(PhantomData<F>);

#[cfg(not(bootstrap))]
impl<F> FmtFnBuilder<F> {
    const INSTANCE: Self = Self(PhantomData);

    #[inline(always)]
    pub const fn offset_argument_ptr<const N: isize>(
        self,
    ) -> FmtFnBuilder<OffsetArgumentPtr<N, F>> {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn offset_literal_ptr<const N: isize>(self) -> FmtFnBuilder<OffsetLiteralPtr<N, F>> {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn write_str(self) -> FmtFnBuilder<WriteStr<F>> {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn set_flags<const FLAGS: u32>(self) -> FmtFnBuilder<SetFlags<FLAGS, F>> {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn set_fill<const FILL: char>(self) -> FmtFnBuilder<SetFill<FILL, F>> {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn clear_align(self) -> FmtFnBuilder<ClearAlign<F>> {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn set_left_align(self) -> FmtFnBuilder<SetLeftAlign<F>> {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn set_right_align(self) -> FmtFnBuilder<SetRightAlign<F>> {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn set_center_align(self) -> FmtFnBuilder<SetCenterAlign<F>> {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn clear_width(self) -> FmtFnBuilder<ClearWidth<F>> {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn set_width<const W: usize>(self) -> FmtFnBuilder<SetWidth<W, F>> {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn set_dyn_width(self) -> FmtFnBuilder<SetDynWidth<F>> {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn clear_precision(self) -> FmtFnBuilder<ClearPrecision<F>> {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn set_precision<const P: usize>(self) -> FmtFnBuilder<SetPrecision<P, F>> {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn set_dyn_precision(self) -> FmtFnBuilder<SetDynPrecision<F>> {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn fmt_binary<T>(self, _: &T) -> FmtFnBuilder<FmtBinary<T, F>>
    where
        T: ?Sized,
    {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn fmt_debug<T>(self, _: &T) -> FmtFnBuilder<FmtDebug<T, F>>
    where
        T: ?Sized,
    {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn fmt_debug_noop<T>(self, _: &T) -> Self
    where
        T: ?Sized,
    {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn fmt_display<T>(self, _: &T) -> FmtFnBuilder<FmtDisplay<T, F>>
    where
        T: ?Sized,
    {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn fmt_lower_exp<T>(self, _: &T) -> FmtFnBuilder<FmtLowerExp<T, F>>
    where
        T: ?Sized,
    {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn fmt_lower_hex<T>(self, _: &T) -> FmtFnBuilder<FmtLowerHex<T, F>>
    where
        T: ?Sized,
    {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn fmt_octal<T>(self, _: &T) -> FmtFnBuilder<FmtOctal<T, F>>
    where
        T: ?Sized,
    {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn fmt_pointer<T>(self, _: &T) -> FmtFnBuilder<FmtPointer<T, F>>
    where
        T: ?Sized,
    {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn fmt_upper_exp<T>(self, _: &T) -> FmtFnBuilder<FmtUpperExp<T, F>>
    where
        T: ?Sized,
    {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn fmt_upper_hex<T>(self, _: &T) -> FmtFnBuilder<FmtUpperHex<T, F>>
    where
        T: ?Sized,
    {
        FmtFnBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn build(self) -> FmtFn
    where
        F: FmtOp,
    {
        F::fmt
    }
}

#[cfg(not(bootstrap))]
impl FmtFnBuilder<End> {
    pub const fn new() -> Self {
        Self::INSTANCE
    }
}

#[cfg(not(bootstrap))]
#[lang = "format_argument_checker"]
pub struct ArgumentChecker(!);

#[cfg(not(bootstrap))]
#[rustc_diagnostic_item = "ArgumentCheckerMethods"]
impl ArgumentChecker {
    #[inline(always)]
    pub fn assert_binary<T>(value: &T) -> &T
    where
        T: Binary,
    {
        value
    }

    #[inline(always)]
    pub fn assert_debug<T>(value: &T) -> &T
    where
        T: Debug,
    {
        value
    }

    #[inline(always)]
    pub fn assert_display<T>(value: &T) -> &T
    where
        T: Display,
    {
        value
    }

    #[inline(always)]
    pub fn assert_lower_exp<T>(value: &T) -> &T
    where
        T: LowerExp,
    {
        value
    }

    #[inline(always)]
    pub fn assert_lower_hex<T>(value: &T) -> &T
    where
        T: LowerHex,
    {
        value
    }

    #[inline(always)]
    pub fn assert_octal<T>(value: &T) -> &T
    where
        T: Octal,
    {
        value
    }

    #[inline(always)]
    pub fn assert_pointer<T>(value: &T) -> &T
    where
        T: Pointer,
    {
        value
    }

    #[inline(always)]
    pub fn assert_upper_exp<T>(value: &T) -> &T
    where
        T: UpperExp,
    {
        value
    }

    #[inline(always)]
    pub fn assert_upper_hex<T>(value: &T) -> &T
    where
        T: UpperHex,
    {
        value
    }
}

/// Format data that can be determined at compile time.
#[cfg(not(bootstrap))]
#[lang = "format_compile_time_data"]
#[repr(C)]
pub struct CompileTimeData<const N: usize> {
    pub literals: [&'static str; N],
}

#[cfg(not(bootstrap))]
impl<const N: usize> CompileTimeData<N> {
    #[inline(always)]
    pub const fn new(literals: [&'static str; N]) -> Self {
        Self { literals }
    }
}

/// This struct represents the unsafety of constructing an `Arguments`.
/// It exists, rather than an unsafe function, in order to simplify the expansion
/// of `format_args!(..)` and reduce the scope of the `unsafe` block.
#[lang = "format_unsafe_arg"]
pub struct UnsafeArg {
    _private: (),
}

impl UnsafeArg {
    /// See documentation where `UnsafeArg` is required to know when it is safe to
    /// create and use `UnsafeArg`.
    #[inline]
    pub const unsafe fn new() -> Self {
        Self { _private: () }
    }
}
