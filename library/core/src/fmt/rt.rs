#![allow(missing_debug_implementations)]
#![unstable(feature = "fmt_internals", reason = "internal to format_args!", issue = "none")]

//! These are the lang items used by format_args!().

use crate::fmt::{
    self, Binary, Debug, Display, Formatter, LowerExp, LowerHex, Octal, Pointer, UpperExp, UpperHex,
};
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
union ArgumentData {
    // Points to one of:
    //
    // - `ArgumentsVTable`: Function pointers to define behavior of `fmt::Arguments`.
    // - `[&'static str]`: All literal pieces used in `format_args`.
    // - An argument used in placeholders.
    ptr: NonNull<()>,

    // Width or precision value specified at runtime.
    usize: usize,
}

#[cfg(not(bootstrap))]
#[lang = "format_argument"]
pub struct Argument<'a> {
    data: ArgumentData,
    _phantom_data: PhantomData<&'a ()>,
}

#[cfg(not(bootstrap))]
#[rustc_diagnostic_item = "ArgumentMethods"]
impl<'a> Argument<'a> {
    #[inline(always)]
    pub const fn from_ref<T>(value: &'a T) -> Self
    where
        T: ?Sized,
    {
        Self {
            data: ArgumentData { ptr: NonNull::from_ref(value).cast() },
            _phantom_data: PhantomData,
        }
    }

    #[inline(always)]
    pub const fn from_str_array(str_array: &'static [&'static str]) -> Self {
        Self::from_ref(str_array)
    }

    #[inline(always)]
    pub fn from_usize(value: &usize) -> Self {
        Self { data: ArgumentData { usize: *value }, _phantom_data: PhantomData }
    }

    /// # Safety
    ///
    /// This object is created with `Argument::from_ref` or `Argument::from_str_array`.
    #[inline(always)]
    pub unsafe fn as_ptr<T>(&self) -> NonNull<T> {
        // SAFETY: Guaranteed by caller.
        unsafe { self.data.ptr.cast() }
    }

    /// # Safety
    ///
    /// - This object is created with `Argument::from_ref` or `Argument::from_str_array`.
    /// - If this object is created with `Argument::from_ref`, `T` should be the same type that
    ///   this object is created from.
    /// - If this object is created with `Argument::from_str_array`, `T` should be `&str`.
    #[inline(always)]
    pub unsafe fn as_ref<T>(&self) -> &T {
        // SAFETY: Guaranteed by caller.
        unsafe { self.as_ptr().as_ref() }
    }

    /// # Safety
    ///
    /// - This object is created with `Argument::from_usize`.
    #[inline(always)]
    pub unsafe fn as_usize(&self) -> usize {
        // SAFETY: Guaranteed by caller.
        unsafe { self.data.usize }
    }

    #[inline(always)]
    pub const fn noop() {}
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

/// Function pointers to define behavior of `fmt::Arguments`.
#[cfg(not(bootstrap))]
pub struct ArgumentsVTable {
    /// Writes `arguments` into `f`.
    pub fmt: unsafe fn(arguments: NonNull<Argument<'_>>, f: &mut Formatter<'_>) -> fmt::Result,

    /// Calculates an estimated total output size.
    pub estimated_capacity: unsafe fn(arguments: NonNull<Argument<'_>>) -> usize,
}

#[cfg(not(bootstrap))]
enum Never {}

/// Like `PhantomData`, but is not intended for instantiating.
#[cfg(not(bootstrap))]
struct NeverPhantomData<T>
where
    T: ?Sized,
{
    _never: Never,
    _phantom_data: PhantomData<T>,
}

/// A helper trait that is used for building `ArgumentsVTable`, it provides a interface for
/// defining the behavior of `fmt::Arguments`.
#[cfg(not(bootstrap))]
pub trait FmtOp {
    const STARTS_WITH_PLACEHOLDER: bool;
    const HAS_PLACEHOLDER: bool;

    unsafe fn fmt(
        literal: NonNull<&'static str>,
        argument: NonNull<Argument<'_>>,
        f: &mut Formatter<'_>,
    ) -> fmt::Result;

    unsafe fn accumulate_pieces_length(
        literal: NonNull<&'static str>,
        pieces_length: usize,
    ) -> usize;
}

/// Marks the end of a format operation sequence.
#[cfg(not(bootstrap))]
pub struct End(NeverPhantomData<()>);

#[cfg(not(bootstrap))]
pub struct OffsetLiteralPtr<const N: isize, R>(NeverPhantomData<R>);

#[cfg(not(bootstrap))]
impl<const N: isize, R> FmtOp for OffsetLiteralPtr<N, R>
where
    R: FmtOp,
{
    const STARTS_WITH_PLACEHOLDER: bool = false;
    const HAS_PLACEHOLDER: bool = R::HAS_PLACEHOLDER;

    /// # Safety
    ///
    /// `literal.offset(N)` should satisfy the safety requirement of `R::fmt`.
    unsafe fn fmt(
        literal: NonNull<&'static str>,
        argument: NonNull<Argument<'_>>,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        // SAFETY: Guaranteed by caller.
        unsafe { R::fmt(literal.offset(N), argument, f) }
    }

    /// # Safety
    ///
    /// `literal.offset(N)` should satisfy the safety requirement of `R::fmt`.
    unsafe fn accumulate_pieces_length(
        literal: NonNull<&'static str>,
        pieces_length: usize,
    ) -> usize {
        // SAFETY: Guaranteed by caller.
        unsafe { R::accumulate_pieces_length(literal.offset(N), pieces_length) }
    }
}

#[cfg(not(bootstrap))]
pub struct OffsetArgumentPtr<const N: isize, R>(NeverPhantomData<R>);

#[cfg(not(bootstrap))]
impl<const N: isize, R> FmtOp for OffsetArgumentPtr<N, R>
where
    R: FmtOp,
{
    const STARTS_WITH_PLACEHOLDER: bool = true;
    const HAS_PLACEHOLDER: bool = true;

    /// # Safety
    ///
    /// `data.offset(N)` should satisfy the safety requirement of `R::fmt`.
    unsafe fn fmt(
        literal: NonNull<&'static str>,
        argument: NonNull<Argument<'_>>,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        // SAFETY: Guaranteed by caller.
        unsafe { R::fmt(literal, argument.offset(N), f) }
    }

    /// # Safety
    ///
    /// `data.offset(N)` should satisfy the safety requirement of `R::fmt`.
    unsafe fn accumulate_pieces_length(
        literal: NonNull<&'static str>,
        pieces_length: usize,
    ) -> usize {
        // SAFETY: Guaranteed by caller.
        unsafe { R::accumulate_pieces_length(literal, pieces_length) }
    }
}

#[cfg(not(bootstrap))]
pub struct WriteStr<R>(NeverPhantomData<R>);

#[cfg(not(bootstrap))]
impl<R> FmtOp for WriteStr<R>
where
    R: FmtOp,
{
    /// Whether this operation starts with a placeholder operation.
    const STARTS_WITH_PLACEHOLDER: bool = false;

    /// Whether this operation contains a placeholder operation.
    const HAS_PLACEHOLDER: bool = R::HAS_PLACEHOLDER;

    /// # Safety
    ///
    /// - `literal` should point to a valid `&'static str` object.
    /// - It is safe to call `R::fmt(literal, argument, f)`.
    unsafe fn fmt(
        literal: NonNull<&'static str>,
        argument: NonNull<Argument<'_>>,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        // SAFETY: Guaranteed by caller.
        unsafe {
            f.write_str(*literal.as_ref())?;

            R::fmt(literal, argument, f)
        }
    }

    /// # Safety
    ///
    /// - `literal` should point to a valid `&'static str` object.
    /// - `literal` should satisfy the safety requirement of `R::accumulate_pieces_length`.
    unsafe fn accumulate_pieces_length(
        literal: NonNull<&'static str>,
        pieces_length: usize,
    ) -> usize {
        // SAFETY: Guaranteed by caller.
        unsafe { R::accumulate_pieces_length(literal, pieces_length + literal.as_ref().len()) }
    }
}

#[cfg(not(bootstrap))]
impl FmtOp for WriteStr<End> {
    /// Whether this operation starts with a placeholder operation.
    const STARTS_WITH_PLACEHOLDER: bool = false;

    /// Whether this operation contains a placeholder operation.
    const HAS_PLACEHOLDER: bool = false;

    /// # Safety
    ///
    /// - `literal` should point to a valid `&'static str` object.
    unsafe fn fmt(
        literal: NonNull<&'static str>,
        _argument: NonNull<Argument<'_>>,
        f: &mut Formatter<'_>,
    ) -> fmt::Result {
        // SAFETY: Guaranteed by caller.
        unsafe { f.write_str(*literal.as_ref()) }
    }

    /// # Safety
    ///
    /// - `literal` should point to a valid `&'static str` object.
    unsafe fn accumulate_pieces_length(
        literal: NonNull<&'static str>,
        pieces_length: usize,
    ) -> usize {
        // SAFETY: Guaranteed by caller.
        unsafe { pieces_length + literal.as_ref().len() }
    }
}

#[cfg(not(bootstrap))]
macro_rules! define_formatter_options_ops {
    (
        $argument:ident,
        $(
            #[fmt_safety_doc = $fmt_safety_doc:literal]
            $name:ident $(<const $const_generic:ident: $const_generic_type:ty>)?
            =>
            $field:ident = $value:expr,
        )*
    ) => {
        $(
            pub struct $name<$(const $const_generic: $const_generic_type,)? R>(NeverPhantomData<R>);

            impl <$(const $const_generic: $const_generic_type,)? R> FmtOp for $name<$($const_generic,)* R>
            where
                R: FmtOp,
            {
                const STARTS_WITH_PLACEHOLDER: bool = true;
                const HAS_PLACEHOLDER: bool = true;

                /// # Safety
                ///
                #[doc = $fmt_safety_doc]
                unsafe fn fmt(
                    literal: NonNull<&'static str>,
                    $argument: NonNull<Argument<'_>>,
                    f: &mut Formatter<'_>,
                ) -> fmt::Result {
                    // SAFETY: Guaranteed by caller.
                    unsafe {
                        f.options.$field = $value;

                        R::fmt(literal, $argument, f)
                    }
                }

                /// # Safety
                ///
                /// `literal` should satisfy the safety requirement of `R::accumulate_pieces_length`.
                unsafe fn accumulate_pieces_length(
                    literal: NonNull<&'static str>,
                    pieces_length: usize,
                ) -> usize {
                    // SAFETY: Guaranteed by caller.
                    unsafe { R::accumulate_pieces_length(literal, pieces_length) }
                }
            }
        )*
    };
}

#[cfg(not(bootstrap))]
define_formatter_options_ops![
    argument,

    #[fmt_safety_doc = "It is safe to call `R::fmt(literal, $argument, f)`."]
    SetFlags<const F: u32>       => flags = F,

    #[fmt_safety_doc = "It is safe to call `R::fmt(literal, $argument, f)`."]
    SetFill<const F: char>       => fill = F,

    #[fmt_safety_doc = "It is safe to call `R::fmt(literal, $argument, f)`."]
    ClearAlign                   => align = None,

    #[fmt_safety_doc = "It is safe to call `R::fmt(literal, $argument, f)`."]
    SetLeftAlign                 => align = Some(fmt::Alignment::Left),

    #[fmt_safety_doc = "It is safe to call `R::fmt(literal, $argument, f)`."]
    SetRightAlign                => align = Some(fmt::Alignment::Right),

    #[fmt_safety_doc = "It is safe to call `R::fmt(literal, $argument, f)`."]
    SetCenterAlign               => align = Some(fmt::Alignment::Center),

    #[fmt_safety_doc = "It is safe to call `R::fmt(literal, $argument, f)`."]
    ClearWidth                   => width = None,

    #[fmt_safety_doc = "It is safe to call `R::fmt(literal, $argument, f)`."]
    SetWidth<const W: usize>     => width = Some(W),

    #[fmt_safety_doc = "`argument` should point to a valid `usize` object and it is safe to call `R::fmt(literal, $argument, f)`."]
    SetDynWidth                  => width = Some(argument.as_ref().as_usize()),

    #[fmt_safety_doc = "It is safe to call `R::fmt(literal, $argument, f)`."]
    ClearPrecision               => precision = None,

    #[fmt_safety_doc = "It is safe to call `R::fmt(literal, $argument, f)`."]
    SetPrecision<const P: usize> => precision = Some(P),

    #[fmt_safety_doc = "`argument` should point to a valid `usize` object and it is safe to call `R::fmt(literal, $argument, f)`."]
    SetDynPrecision              => precision = Some(argument.as_ref().as_usize()),
];

#[cfg(not(bootstrap))]
macro_rules! define_fmt_trait_ops {
    ($($name:ident => $trait:ident,)*) => {
        $(
            pub struct $name<T, R>(NeverPhantomData<(T, R)>);

            impl<T, R> FmtOp for $name<T, R>
            where
                T: $trait,
                R: FmtOp,
            {
                const STARTS_WITH_PLACEHOLDER: bool = true;
                const HAS_PLACEHOLDER: bool = true;

                /// # Safety
                ///
                /// - `data` should point to a valid `&T` object.
                /// - It is safe to call `R::fmt(literal, $argument, f)`.
                unsafe fn fmt(
                    literal: NonNull<&'static str>,
                    argument: NonNull<Argument<'_>>,
                    f: &mut Formatter<'_>,
                ) -> fmt::Result {
                    // SAFETY: Guaranteed by caller.
                    unsafe {
                        T::fmt(argument.as_ref().as_ref::<T>(), f)?;

                        R::fmt(literal, argument, f)
                    }
                }

                /// # Safety
                ///
                /// `literal` should satisfy the safety requirement of `R::accumulate_pieces_length`.
                unsafe fn accumulate_pieces_length(
                    literal: NonNull<&'static str>,
                    pieces_length: usize,
                ) -> usize {
                    // SAFETY: Guaranteed by caller.
                    unsafe { R::accumulate_pieces_length(literal, pieces_length) }
                }
            }

            impl<T> FmtOp for $name<T, End>
            where
                T: $trait,
            {
                const STARTS_WITH_PLACEHOLDER: bool = true;
                const HAS_PLACEHOLDER: bool = true;

                /// # Safety
                ///
                /// - `data` should point to a valid `&T` object.
                unsafe fn fmt(
                    _literal: NonNull<&'static str>,
                    argument: NonNull<Argument<'_>>,
                    f: &mut Formatter<'_>,
                ) -> fmt::Result {
                    // SAFETY: Guaranteed by caller.
                    unsafe { T::fmt(argument.as_ref().as_ref::<T>(), f) }
                }

                /// # Safety
                ///
                /// Always safe.
                unsafe fn accumulate_pieces_length(
                    _literal: NonNull<&'static str>,
                    pieces_length: usize,
                ) -> usize {
                    // SAFETY: Guaranteed by caller.
                    pieces_length
                }
            }
        )*
    };
}

#[cfg(not(bootstrap))]
define_fmt_trait_ops![
    FmtBinary   => Binary,
    FmtDebug    => Debug,
    FmtDisplay  => Display,
    FmtLowerExp => LowerExp,
    FmtLowerHex => LowerHex,
    FmtOctal    => Octal,
    FmtPointer  => Pointer,
    FmtUpperExp => UpperExp,
    FmtUpperHex => UpperHex,
];

/// A helper type for building `fmt::ArgumentsVTable`.
#[cfg(not(bootstrap))]
pub struct ArgumentsVTableBuilder<R> {
    inner: PhantomData<R>,
}

#[cfg(not(bootstrap))]
#[rustc_diagnostic_item = "ArgumentsVTableBuilderMethods"]
impl<R> ArgumentsVTableBuilder<R> {
    pub const INSTANCE: Self = Self { inner: PhantomData };

    #[inline(always)]
    pub const fn prepend_offset_argument_ptr<const N: isize>(
        self,
    ) -> ArgumentsVTableBuilder<OffsetArgumentPtr<N, R>> {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn prepend_offset_literal_ptr<const N: isize>(
        self,
    ) -> ArgumentsVTableBuilder<OffsetLiteralPtr<N, R>> {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn prepend_write_str(self) -> ArgumentsVTableBuilder<WriteStr<R>> {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn prepend_set_flags<const F: u32>(self) -> ArgumentsVTableBuilder<SetFlags<F, R>> {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn prepend_set_fill<const F: char>(self) -> ArgumentsVTableBuilder<SetFill<F, R>> {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn prepend_clear_align(self) -> ArgumentsVTableBuilder<ClearAlign<R>> {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn prepend_set_left_align(self) -> ArgumentsVTableBuilder<SetLeftAlign<R>> {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn prepend_set_right_align(self) -> ArgumentsVTableBuilder<SetRightAlign<R>> {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn prepend_set_center_align(self) -> ArgumentsVTableBuilder<SetCenterAlign<R>> {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn prepend_clear_width(self) -> ArgumentsVTableBuilder<ClearWidth<R>> {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn prepend_set_width<const W: usize>(self) -> ArgumentsVTableBuilder<SetWidth<W, R>> {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn prepend_set_dyn_width(self) -> ArgumentsVTableBuilder<SetDynWidth<R>> {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn prepend_clear_precision(self) -> ArgumentsVTableBuilder<ClearPrecision<R>> {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn prepend_set_precision<const P: usize>(
        self,
    ) -> ArgumentsVTableBuilder<SetPrecision<P, R>> {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn prepend_set_dyn_precision(self) -> ArgumentsVTableBuilder<SetDynPrecision<R>> {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub fn prepend_fmt_binary<T>(self, _: &T) -> ArgumentsVTableBuilder<FmtBinary<T, R>>
    where
        T: Binary,
    {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub fn prepend_fmt_debug<T>(self, _: &T) -> ArgumentsVTableBuilder<FmtDebug<T, R>>
    where
        T: Debug,
    {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub fn prepend_fmt_debug_noop<T>(self, _: &T) -> Self
    where
        T: Debug,
    {
        self
    }

    #[inline(always)]
    pub fn prepend_fmt_display<T>(self, _: &T) -> ArgumentsVTableBuilder<FmtDisplay<T, R>>
    where
        T: Display,
    {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub fn prepend_fmt_lower_exp<T>(self, _: &T) -> ArgumentsVTableBuilder<FmtLowerExp<T, R>>
    where
        T: LowerExp,
    {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub fn prepend_fmt_lower_hex<T>(self, _: &T) -> ArgumentsVTableBuilder<FmtLowerHex<T, R>>
    where
        T: LowerHex,
    {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub fn prepend_fmt_octal<T>(self, _: &T) -> ArgumentsVTableBuilder<FmtOctal<T, R>>
    where
        T: Octal,
    {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub fn prepend_fmt_pointer<T>(self, _: &T) -> ArgumentsVTableBuilder<FmtPointer<T, R>>
    where
        T: Pointer,
    {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub fn prepend_fmt_upper_exp<T>(self, _: &T) -> ArgumentsVTableBuilder<FmtUpperExp<T, R>>
    where
        T: UpperExp,
    {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub fn prepend_fmt_upper_hex<T>(self, _: &T) -> ArgumentsVTableBuilder<FmtUpperHex<T, R>>
    where
        T: UpperHex,
    {
        ArgumentsVTableBuilder::INSTANCE
    }

    #[inline(always)]
    pub const fn finish(self) -> &'static ArgumentsVTable
    where
        R: FmtOp,
    {
        /// # Safety
        ///
        /// The memory pointed by `arguments` should match `R::fmt`’s memory access behavior.
        unsafe fn fmt<R>(arguments: NonNull<Argument<'_>>, f: &mut Formatter<'_>) -> fmt::Result
        where
            R: FmtOp,
        {
            // SAFETY: Guaranteed by caller.
            unsafe { R::fmt(arguments.as_ref().as_ptr::<&'static str>(), arguments, f) }
        }

        /// # Safety
        ///
        /// The memory pointed by `arguments` should match `R::accumulate_pieces_length`’s memory access behavior.
        unsafe fn estimated_capacity<R>(arguments: NonNull<Argument<'_>>) -> usize
        where
            R: FmtOp,
        {
            // SAFETY: Guaranteed by caller.
            let pieces_length = unsafe {
                R::accumulate_pieces_length(arguments.as_ref().as_ptr::<&'static str>(), 0)
            };

            if R::HAS_PLACEHOLDER {
                if R::STARTS_WITH_PLACEHOLDER && pieces_length < 16 {
                    // If the format string starts with an argument,
                    // don't preallocate anything, unless length
                    // of pieces is significant.
                    0
                } else {
                    // There are some arguments, so any additional push
                    // will reallocate the string. To avoid that,
                    // we're "pre-doubling" the capacity here.
                    pieces_length.checked_mul(2).unwrap_or(0)
                }
            } else {
                pieces_length
            }
        }

        &ArgumentsVTable { fmt: fmt::<R>, estimated_capacity: estimated_capacity::<R> }
    }
}
