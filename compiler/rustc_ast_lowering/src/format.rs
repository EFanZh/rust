use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::{iter, slice};

use rustc_ast::*;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_session::config::FmtDebug;
use rustc_span::{Ident, Span, Symbol, kw, sym};

use super::LoweringContext;

impl<'hir> LoweringContext<'_, 'hir> {
    pub(crate) fn lower_format_args(&mut self, sp: Span, fmt: &FormatArgs) -> hir::ExprKind<'hir> {
        // Never call the const constructor of `fmt::Arguments` if the
        // format_args!() had any arguments _before_ flattening/inlining.
        let allow_const = fmt.arguments.all_args().is_empty();
        let mut fmt = Cow::Borrowed(fmt);
        if self.tcx.sess.opts.unstable_opts.flatten_format_args {
            fmt = flatten_format_args(fmt);
            fmt = self.inline_literals(fmt);
        }
        expand_format_args(self, sp, &fmt, allow_const)
    }

    /// Try to convert a literal into an interned string
    fn try_inline_lit(&self, lit: token::Lit) -> Option<Symbol> {
        match LitKind::from_token_lit(lit) {
            Ok(LitKind::Str(s, _)) => Some(s),
            Ok(LitKind::Int(n, ty)) => {
                match ty {
                    // unsuffixed integer literals are assumed to be i32's
                    LitIntType::Unsuffixed => {
                        (n <= i32::MAX as u128).then_some(Symbol::intern(&n.to_string()))
                    }
                    LitIntType::Signed(int_ty) => {
                        let max_literal = self.int_ty_max(int_ty);
                        (n <= max_literal).then_some(Symbol::intern(&n.to_string()))
                    }
                    LitIntType::Unsigned(uint_ty) => {
                        let max_literal = self.uint_ty_max(uint_ty);
                        (n <= max_literal).then_some(Symbol::intern(&n.to_string()))
                    }
                }
            }
            _ => None,
        }
    }

    /// Get the maximum value of int_ty. It is platform-dependent due to the byte size of isize
    fn int_ty_max(&self, int_ty: IntTy) -> u128 {
        match int_ty {
            IntTy::Isize => self.tcx.data_layout.pointer_size.signed_int_max() as u128,
            IntTy::I8 => i8::MAX as u128,
            IntTy::I16 => i16::MAX as u128,
            IntTy::I32 => i32::MAX as u128,
            IntTy::I64 => i64::MAX as u128,
            IntTy::I128 => i128::MAX as u128,
        }
    }

    /// Get the maximum value of uint_ty. It is platform-dependent due to the byte size of usize
    fn uint_ty_max(&self, uint_ty: UintTy) -> u128 {
        match uint_ty {
            UintTy::Usize => self.tcx.data_layout.pointer_size.unsigned_int_max(),
            UintTy::U8 => u8::MAX as u128,
            UintTy::U16 => u16::MAX as u128,
            UintTy::U32 => u32::MAX as u128,
            UintTy::U64 => u64::MAX as u128,
            UintTy::U128 => u128::MAX as u128,
        }
    }

    /// Inline literals into the format string.
    ///
    /// Turns
    ///
    /// `format_args!("Hello, {}! {} {}", "World", 123, x)`
    ///
    /// into
    ///
    /// `format_args!("Hello, World! 123 {}", x)`.
    fn inline_literals<'fmt>(&self, mut fmt: Cow<'fmt, FormatArgs>) -> Cow<'fmt, FormatArgs> {
        let mut was_inlined = vec![false; fmt.arguments.all_args().len()];
        let mut inlined_anything = false;

        for i in 0..fmt.template.len() {
            let FormatArgsPiece::Placeholder(placeholder) = &fmt.template[i] else { continue };
            let Ok(arg_index) = placeholder.argument.index else { continue };

            let mut literal = None;

            if let FormatTrait::Display = placeholder.format_trait
                && placeholder.format_options == Default::default()
                && let arg = fmt.arguments.all_args()[arg_index].expr.peel_parens_and_refs()
                && let ExprKind::Lit(lit) = arg.kind
            {
                literal = self.try_inline_lit(lit);
            }

            if let Some(literal) = literal {
                // Now we need to mutate the outer FormatArgs.
                // If this is the first time, this clones the outer FormatArgs.
                let fmt = fmt.to_mut();
                // Replace the placeholder with the literal.
                fmt.template[i] = FormatArgsPiece::Literal(literal);
                was_inlined[arg_index] = true;
                inlined_anything = true;
            }
        }

        // Remove the arguments that were inlined.
        if inlined_anything {
            let fmt = fmt.to_mut();

            let mut remove = was_inlined;

            // Don't remove anything that's still used.
            for_all_argument_indexes(&mut fmt.template, |index| remove[*index] = false);

            // Drop all the arguments that are marked for removal.
            let mut remove_it = remove.iter();
            fmt.arguments.all_args_mut().retain(|_| remove_it.next() != Some(&true));

            // Calculate the mapping of old to new indexes for the remaining arguments.
            let index_map: Vec<usize> = remove
                .into_iter()
                .scan(0, |i, remove| {
                    let mapped = *i;
                    *i += !remove as usize;
                    Some(mapped)
                })
                .collect();

            // Correct the indexes that refer to arguments that have shifted position.
            for_all_argument_indexes(&mut fmt.template, |index| *index = index_map[*index]);
        }

        fmt
    }
}

/// Flattens nested `format_args!()` into one.
///
/// Turns
///
/// `format_args!("a {} {} {}.", 1, format_args!("b{}!", 2), 3)`
///
/// into
///
/// `format_args!("a {} b{}! {}.", 1, 2, 3)`.
fn flatten_format_args(mut fmt: Cow<'_, FormatArgs>) -> Cow<'_, FormatArgs> {
    let mut i = 0;
    while i < fmt.template.len() {
        if let FormatArgsPiece::Placeholder(placeholder) = &fmt.template[i]
            && let FormatTrait::Display | FormatTrait::Debug = &placeholder.format_trait
            && let Ok(arg_index) = placeholder.argument.index
            && let arg = fmt.arguments.all_args()[arg_index].expr.peel_parens_and_refs()
            && let ExprKind::FormatArgs(_) = &arg.kind
            // Check that this argument is not used by any other placeholders.
            && fmt.template.iter().enumerate().all(|(j, p)|
                i == j ||
                !matches!(p, FormatArgsPiece::Placeholder(placeholder)
                    if placeholder.argument.index == Ok(arg_index))
            )
        {
            // Now we need to mutate the outer FormatArgs.
            // If this is the first time, this clones the outer FormatArgs.
            let fmt = fmt.to_mut();

            // Take the inner FormatArgs out of the outer arguments, and
            // replace it by the inner arguments. (We can't just put those at
            // the end, because we need to preserve the order of evaluation.)

            let args = fmt.arguments.all_args_mut();
            let remaining_args = args.split_off(arg_index + 1);
            let old_arg_offset = args.len();
            let mut fmt2 = &mut args.pop().unwrap().expr; // The inner FormatArgs.
            let fmt2 = loop {
                // Unwrap the Expr to get to the FormatArgs.
                match &mut fmt2.kind {
                    ExprKind::Paren(inner) | ExprKind::AddrOf(BorrowKind::Ref, _, inner) => {
                        fmt2 = inner
                    }
                    ExprKind::FormatArgs(fmt2) => break fmt2,
                    _ => unreachable!(),
                }
            };

            args.append(fmt2.arguments.all_args_mut());
            let new_arg_offset = args.len();
            args.extend(remaining_args);

            // Correct the indexes that refer to the arguments after the newly inserted arguments.
            for_all_argument_indexes(&mut fmt.template, |index| {
                if *index >= old_arg_offset {
                    *index -= old_arg_offset;
                    *index += new_arg_offset;
                }
            });

            // Now merge the placeholders:

            let rest = fmt.template.split_off(i + 1);
            fmt.template.pop(); // remove the placeholder for the nested fmt args.
            // Insert the pieces from the nested format args, but correct any
            // placeholders to point to the correct argument index.
            for_all_argument_indexes(&mut fmt2.template, |index| *index += arg_index);
            fmt.template.append(&mut fmt2.template);
            fmt.template.extend(rest);

            // Don't increment `i` here, so we recurse into the newly added pieces.
        } else {
            i += 1;
        }
    }
    fmt
}

enum ArgumentUsage {
    Ref,
    Usize(Option<Span>),
}

enum ArgumentRefCount {
    Zero,
    One(Span),
    More,
}

impl ArgumentRefCount {
    fn add_ref(&mut self, span: Option<Span>) {
        if let Some(span) = span {
            *self = match self {
                Self::Zero => Self::One(span),
                Self::One(_) => Self::More,
                Self::More => return,
            };
        }
    }
}

/// Used for comparing whether two format count refers to the same value.
#[derive(Clone, Copy, Eq, PartialEq)]
enum FormatCountKey {
    Literal(usize),
    Argument(usize),
}

impl<'a> TryFrom<&'a FormatCount> for FormatCountKey {
    type Error = ();

    fn try_from(value: &'a FormatCount) -> Result<Self, Self::Error> {
        match value {
            FormatCount::Literal(count) => Ok(Self::Literal(*count)),
            FormatCount::Argument(format_arg_position) => match format_arg_position.index {
                Ok(index) => Ok(Self::Argument(index)),
                Err(_) => Err(()),
            },
        }
    }
}

enum OffsetPtrKind {
    Literal,
    Argument,
}

enum FormatOp<'a> {
    OffsetPtr(isize, OffsetPtrKind),
    WriteStr,
    SetFlags(u32),
    SetFill(char),
    ClearAlign,
    SetLeftAlign,
    SetRightAlign,
    SetCenterAlign,
    ClearWidth,
    SetWidth(usize),
    SetDynWidth,
    ClearPrecision,
    SetPrecision(usize),
    SetDynPrecision,
    FmtValue(&'a FormatPlaceholder),
}

struct FormatOpsEncoder<'a> {
    // Literal pieces used in `format_args`.
    literal_data: Vec<Symbol>,

    // Index of each literal piece in `literal_data`.
    literal_offsets: FxHashMap<Symbol, usize>,

    // Runtime arguments used in `format_args`.
    argument_data: Vec<(usize, ArgumentUsage)>,

    // Index of each argument that is used as formatting value in `argument_data`.
    ref_argument_offsets: Box<[usize]>,

    // Index of each argument that is used as width or precision in `argument_data`.
    usize_argument_offsets: FxHashMap<usize, usize>,

    argument_ref_counts: Box<[ArgumentRefCount]>,

    current_literal_offset: usize,
    current_argument_offset: usize,
    current_flags: u32,
    current_fill: char,
    current_align: Option<FormatAlignment>,
    current_width: Option<FormatCountKey>,
    current_precision: Option<FormatCountKey>,
    incomplete_literal: Vec<Symbol>,
    literal_buffer: String,
    format_ops: Vec<FormatOp<'a>>,
}

impl<'a> FormatOpsEncoder<'a> {
    fn encode_literal(&mut self, literal: Symbol) {
        if literal != kw::Empty {
            self.incomplete_literal.push(literal);
        }
    }

    fn set_literal_offset(&mut self, offset: usize) {
        if self.current_literal_offset != offset {
            let diff = usize::wrapping_sub(offset, self.current_literal_offset) as isize;

            self.current_literal_offset = offset;
            self.format_ops.push(FormatOp::OffsetPtr(diff, OffsetPtrKind::Literal));
        }
    }

    fn set_argument_offset(&mut self, offset: usize) {
        if self.current_argument_offset != offset {
            let diff = usize::wrapping_sub(offset, self.current_argument_offset) as isize;

            self.current_argument_offset = offset;
            self.format_ops.push(FormatOp::OffsetPtr(diff, OffsetPtrKind::Argument));
        }
    }

    fn allocate_literal(&mut self, literal: Symbol) {
        let offset = *self.literal_offsets.entry(literal).or_insert_with(|| {
            let offset = self.literal_data.len();

            self.literal_data.push(literal);

            offset
        });

        self.set_literal_offset(offset);
    }

    fn allocate_ref_argument(&mut self, placeholder: &'a FormatPlaceholder) {
        let index = placeholder.argument.index.unwrap();

        self.argument_ref_counts[index].add_ref(placeholder.span);

        let argument_offset = &mut self.ref_argument_offsets[index];

        if *argument_offset == usize::MAX {
            let offset = self.argument_data.len();

            self.argument_data.push((index, ArgumentUsage::Ref));

            *argument_offset = offset;
        }

        let offset = *argument_offset;

        self.set_argument_offset(offset);
    }

    fn allocate_usize_argument(&mut self, format_arg_position: &FormatArgPosition) {
        let index = format_arg_position.index.unwrap();
        let placeholder_span = format_arg_position.span;

        self.argument_ref_counts[index].add_ref(placeholder_span);

        let offset = match self.usize_argument_offsets.entry(index) {
            Entry::Occupied(occupied_entry) => {
                let offset = *occupied_entry.get();

                let ArgumentUsage::Usize(span) = &mut self.argument_data[offset].1 else {
                    unreachable!()
                };

                if span.is_none() {
                    *span = placeholder_span;
                }

                offset
            }
            Entry::Vacant(vacant_entry) => {
                let offset = *vacant_entry.insert(self.argument_data.len());

                self.argument_data.push((index, ArgumentUsage::Usize(placeholder_span)));

                offset
            }
        };

        self.set_argument_offset(offset);
    }

    fn flush_literal(&mut self) {
        let literal = match &*self.incomplete_literal {
            [] => return,
            &[literal] => literal,
            pieces => {
                self.literal_buffer.reserve(
                    pieces
                        .iter()
                        .try_fold(0, |sum, piece| usize::checked_add(sum, piece.as_str().len()))
                        .unwrap(),
                );

                for piece in pieces {
                    self.literal_buffer.push_str(piece.as_str());
                }

                let literal = Symbol::intern(&self.literal_buffer);

                self.literal_buffer.clear();

                literal
            }
        };

        self.incomplete_literal.clear();

        self.allocate_literal(literal);

        self.format_ops.push(FormatOp::WriteStr);
    }

    fn encode_placeholder(&mut self, placeholder: &'a FormatPlaceholder) {
        self.flush_literal();

        // Update format options that have changed.

        let format_options = &placeholder.format_options;

        // Update flags.

        // This needs to match `Flag` in library/core/src/fmt/rt.rs.
        let flags: u32 = ((format_options.sign == Some(FormatSign::Plus)) as u32)
            | ((format_options.sign == Some(FormatSign::Minus)) as u32) << 1
            | (format_options.alternate as u32) << 2
            | (format_options.zero_pad as u32) << 3
            | ((format_options.debug_hex == Some(FormatDebugHex::Lower)) as u32) << 4
            | ((format_options.debug_hex == Some(FormatDebugHex::Upper)) as u32) << 5;

        if self.current_flags != flags {
            self.current_flags = flags;
            self.format_ops.push(FormatOp::SetFlags(self.current_flags));
        }

        // Update fill.

        let fill = format_options.fill.unwrap_or(' ');

        if self.current_fill != fill {
            self.current_fill = fill;
            self.format_ops.push(FormatOp::SetFill(self.current_fill))
        }

        // Update align.

        let align = format_options.alignment;

        if self.current_align != align {
            self.current_align = align;

            self.format_ops.push(match self.current_align {
                None => FormatOp::ClearAlign,
                Some(FormatAlignment::Left) => FormatOp::SetLeftAlign,
                Some(FormatAlignment::Right) => FormatOp::SetRightAlign,
                Some(FormatAlignment::Center) => FormatOp::SetCenterAlign,
            });
        }

        // Update width.

        let width =
            format_options.width.as_ref().map(FormatCountKey::try_from).transpose().unwrap();

        if self.current_width != width {
            self.current_width = width.clone();

            let update_width_op = match &format_options.width {
                None => FormatOp::ClearWidth,
                Some(FormatCount::Literal(width)) => FormatOp::SetWidth(*width),
                Some(FormatCount::Argument(arg)) => {
                    self.allocate_usize_argument(arg);

                    FormatOp::SetDynWidth
                }
            };

            self.format_ops.push(update_width_op);
        }

        // Update precision.

        let precision =
            format_options.precision.as_ref().map(FormatCountKey::try_from).transpose().unwrap();

        if self.current_precision != precision {
            self.current_precision = precision;

            let update_precision_op = match &format_options.precision {
                None => FormatOp::ClearPrecision,
                Some(FormatCount::Literal(precision)) => FormatOp::SetPrecision(*precision),
                Some(FormatCount::Argument(arg)) => {
                    self.allocate_usize_argument(arg);

                    FormatOp::SetDynPrecision
                }
            };

            self.format_ops.push(update_precision_op);
        }

        // Update argument pointer.

        self.allocate_ref_argument(placeholder);

        self.format_ops.push(FormatOp::FmtValue(placeholder));
    }

    fn finish(&mut self) {
        self.flush_literal();
    }
}

fn make_method_call_expr<'hir>(
    ctx: &mut LoweringContext<'_, 'hir>,
    span: Span,
    object: &'hir hir::Expr<'hir>,
    method: Symbol,
    generic_args: Option<&'hir hir::GenericArgs<'hir>>,
    args: &'hir [hir::Expr<'hir>],
) -> &'hir hir::Expr<'hir> {
    let hir_id = ctx.next_id();

    ctx.arena.alloc(ctx.expr(
        span,
        hir::ExprKind::MethodCall(
            ctx.arena.alloc(hir::PathSegment {
                ident: Ident::new(method, span),
                hir_id,
                res: hir::def::Res::Err,
                args: generic_args,
                infer_args: generic_args.is_none(),
            }),
            object,
            args,
            span,
        ),
    ))
}

fn expand_format_args<'hir>(
    ctx: &mut LoweringContext<'_, 'hir>,
    macsp: Span,
    fmt: &FormatArgs,
    allow_const: bool,
) -> hir::ExprKind<'hir> {
    // Estimate buffer sizes required for `FormatOpsEncoder`.

    let mut consecutive_literal_pieces = 0;
    let mut estimated_literal_count = 0;
    let mut estimated_incomplete_literal_capacity = 0;
    let mut estimated_usize_arguments = 0;
    let mut estimated_format_ops = 0;
    let mut literal_length = Some(0);

    for piece in &fmt.template {
        match piece {
            FormatArgsPiece::Literal(symbol) => {
                if *symbol != kw::Empty {
                    if consecutive_literal_pieces == 0 {
                        estimated_literal_count += 1;
                        estimated_format_ops += 2;
                    }

                    consecutive_literal_pieces += 1;

                    estimated_incomplete_literal_capacity =
                        estimated_incomplete_literal_capacity.max(consecutive_literal_pieces);

                    literal_length = literal_length.and_then(|literal_length| {
                        u64::checked_add(literal_length, symbol.as_str().len() as u64)
                    });
                }
            }
            FormatArgsPiece::Placeholder(placeholder) => {
                consecutive_literal_pieces = 0;

                estimated_usize_arguments += usize::from(matches!(
                    placeholder.format_options.precision,
                    Some(FormatCount::Argument(_))
                ));

                estimated_usize_arguments += usize::from(matches!(
                    placeholder.format_options.width,
                    Some(FormatCount::Argument(_))
                ));

                estimated_format_ops += 2;
            }
        }
    }

    let argument_count = fmt.arguments.all_args().len();

    // Encode format operations.

    let mut format_ops_encoder = FormatOpsEncoder {
        literal_data: Vec::with_capacity(estimated_literal_count),
        literal_offsets: FxHashMap::with_capacity_and_hasher(
            estimated_literal_count,
            Default::default(),
        ),
        argument_data: Vec::with_capacity(argument_count),
        ref_argument_offsets: vec![usize::MAX; argument_count].into_boxed_slice(),
        usize_argument_offsets: FxHashMap::with_capacity_and_hasher(
            estimated_usize_arguments,
            Default::default(),
        ),
        argument_ref_counts: iter::repeat_with(|| ArgumentRefCount::Zero)
            .take(argument_count)
            .collect(),
        current_literal_offset: 0,
        current_argument_offset: 0,
        current_flags: 0,
        current_fill: ' ',
        current_align: None,
        current_width: None,
        current_precision: None,
        incomplete_literal: Vec::with_capacity(estimated_incomplete_literal_capacity),
        literal_buffer: String::new(),
        format_ops: Vec::with_capacity(estimated_format_ops),
    };

    for piece in &fmt.template {
        match piece {
            FormatArgsPiece::Literal(literal) => format_ops_encoder.encode_literal(*literal),
            FormatArgsPiece::Placeholder(placeholder) => {
                format_ops_encoder.encode_placeholder(placeholder);
            }
        }
    }

    format_ops_encoder.finish();

    let (literal_data, argument_data, argument_ref_counts, format_ops) = {
        let format_ops_encoder = format_ops_encoder;

        (
            format_ops_encoder.literal_data,
            format_ops_encoder.argument_data,
            format_ops_encoder.argument_ref_counts,
            format_ops_encoder.format_ops,
        )
    };

    let arena = ctx.arena;

    // Use `<core::fmt::Arguments>::from_static_str(_with_lifetime)` if there is no arguments.

    if fmt.arguments.all_args().is_empty() {
        let literal = match *literal_data {
            [] => kw::Empty,
            [literal] => literal,
            _ => unreachable!(),
        };

        let literal_expr = ctx.expr_str(macsp, literal);

        let (method, args) = if allow_const {
            // Generate:
            //     <core::fmt::Arguments>::from_static_str(literal_expr)

            (sym::from_static_str, slice::from_ref(arena.alloc(literal_expr)))
        } else {
            // Generate:
            //     <core::fmt::Arguments>::from_static_str_with_lifetime(literal_expr, &core::fmt::rt::Argument::noop());

            let argument_noop =
                ctx.expr_lang_item_type_relative(macsp, hir::LangItem::FormatArgument, sym::noop);

            let lifetime_arg_value = ctx.expr_call(macsp, arena.alloc(argument_noop), &[]);

            let lifetime_arg = ctx.expr(
                macsp,
                hir::ExprKind::AddrOf(
                    hir::BorrowKind::Ref,
                    hir::Mutability::Not,
                    lifetime_arg_value,
                ),
            );

            (
                sym::from_static_str_with_lifetime,
                &*arena.alloc_from_iter([literal_expr, lifetime_arg]),
            )
        };

        let arguments_from_str = arena.alloc(ctx.expr_lang_item_type_relative(
            macsp,
            hir::LangItem::FormatArguments,
            method,
        ));

        return hir::ExprKind::Call(arguments_from_str, args);
    }

    // There are some arguments, we will generate step by step:
    //     <core::fmt::Arguments>::new(
    //         &match (&arg0, &arg1, &arg2, …) {
    //             args => [
    //                 <core::fmt::rt::Argument>::from_fmt_fn(
    //                     <core::fmt::rt::FmtFnBuilder>::new()
    //                         …
    //                         .fmt_debug(args.1)
    //                         .offset_argument_ptr::<1>()
    //                         .write_str()
    //                         .offset_literal_ptr::<1>()
    //                         .fmt_display(args.0)
    //                         .write_str()
    //                         .build(),
    //                 ),
    //                 <core::fmt::rt::Argument>::from_compile_time_data(
    //                     &const {
    //                         <core::fmt::rt::CompileTimeData>::new(
    //                             ["…", "…", "…", …],
    //                         )
    //                     }
    //                 ),
    //                 <core::fmt::rt::Argument>::from_ref(<core::fmt::rt::Argument>::assert_…(args.0)),
    //                 <core::fmt::rt::Argument>::from_ref(<core::fmt::rt::Argument>::assert_…(args.1)),
    //                 <core::fmt::rt::Argument>::from_ref(<core::fmt::rt::Argument>::assert_…(args.2)),
    //                 …
    //             ],
    //         },
    //         unsafe { ::core::fmt::rt::UnsafeArg::new() }
    //     )

    // Generate:
    //     (&arg0, &arg1, &arg2, …)

    let arguments = fmt.arguments.all_args();
    let macsp_ctxt = macsp.ctxt();

    let args_tuple_elements =
        arena.alloc_from_iter(arguments.iter().zip(argument_ref_counts).map(|(arg, ref_count)| {
            let arg_span = if let (FormatArgumentKind::Captured(_), ArgumentRefCount::One(span)) =
                (&arg.kind, ref_count)
            {
                span
            } else {
                arg.expr.span
            };

            let arg_span = arg_span.with_ctxt(macsp_ctxt);
            let arg = ctx.lower_expr(&arg.expr);

            ctx.expr(
                arg_span,
                hir::ExprKind::AddrOf(hir::BorrowKind::Ref, hir::Mutability::Not, arg),
            )
        }));

    let args_tuple = ctx.arena.alloc(ctx.expr(macsp, hir::ExprKind::Tup(args_tuple_elements)));

    // Generate:
    //     <core::fmt::rt::from_fmt_fn>

    let argument_from_fmt_fn = arena.alloc(ctx.expr_lang_item_type_relative(
        macsp,
        hir::LangItem::FormatArgument,
        sym::from_fmt_fn,
    ));

    // Generate:
    //     <core::fmt::rt::FmtFnBuilder>::new()
    //         …
    //         .fmt_debug(args.1)
    //         .offset_argument_ptr::<1>()
    //         .write_str()
    //         .offset_literal_ptr::<1>()
    //         .fmt_display(args.0)
    //         .write_str()
    //         .build()

    let args_ident = Ident::new(sym::args, macsp);
    let (args_pat, args_hir_id) = ctx.pat_ident(macsp, args_ident);

    let argument_fmt_fn_builder = arena.alloc(ctx.expr_lang_item_type_relative(
        macsp,
        rustc_hir::LangItem::FormatFmtFnBuilder,
        sym::new,
    ));

    let mut fmt_fn_expr = ctx.expr_call(macsp, argument_fmt_fn_builder, &[]);

    let fmt_debug_method = match ctx.tcx.sess.opts.unstable_opts.fmt_debug {
        FmtDebug::Full | FmtDebug::Shallow => sym::fmt_debug,
        FmtDebug::None => sym::fmt_debug_noop,
    };

    let starts_with_literal = matches!(format_ops.first(), Some(FormatOp::WriteStr));

    // Build format operation in reverse order.

    for format_op in format_ops.into_iter().rev() {
        let (method, generic_arg, arg) = match format_op {
            FormatOp::OffsetPtr(n, kind) => {
                let method = match kind {
                    OffsetPtrKind::Literal => sym::offset_literal_ptr,
                    OffsetPtrKind::Argument => sym::offset_argument_ptr,
                };

                let literal_expr = ctx.expr(
                    macsp,
                    hir::ExprKind::Lit(arena.alloc(hir::Lit {
                        span: macsp,
                        node: LitKind::Int(
                            (n.unsigned_abs() as u128).into(),
                            LitIntType::Signed(IntTy::Isize),
                        ),
                    })),
                );

                let signed_literal_expr = if n < 0 {
                    ctx.expr(macsp, hir::ExprKind::Unary(hir::UnOp::Neg, arena.alloc(literal_expr)))
                } else {
                    literal_expr
                };

                (method, Some(signed_literal_expr), None)
            }
            FormatOp::WriteStr => (sym::write_str, None, None),
            FormatOp::SetFlags(flags) => (sym::set_flags, Some(ctx.expr_u32(macsp, flags)), None),
            FormatOp::SetFill(fill) => (sym::set_fill, Some(ctx.expr_char(macsp, fill)), None),
            FormatOp::ClearAlign => (sym::clear_align, None, None),
            FormatOp::SetLeftAlign => (sym::set_left_align, None, None),
            FormatOp::SetRightAlign => (sym::set_right_align, None, None),
            FormatOp::SetCenterAlign => (sym::set_center_align, None, None),
            FormatOp::ClearWidth => (sym::clear_width, None, None),
            FormatOp::SetWidth(width) => (sym::set_width, Some(ctx.expr_usize(macsp, width)), None),
            FormatOp::SetDynWidth => (sym::set_dyn_width, None, None),
            FormatOp::ClearPrecision => (sym::clear_precision, None, None),
            FormatOp::SetPrecision(precision) => {
                (sym::set_precision, Some(ctx.expr_usize(macsp, precision)), None)
            }
            FormatOp::SetDynPrecision => (sym::set_dyn_precision, None, None),
            FormatOp::FmtValue(placeholder) => {
                let (method, assert_method) = match placeholder.format_trait {
                    FormatTrait::Display => (sym::fmt_display, sym::assert_display),
                    FormatTrait::Debug => (fmt_debug_method, sym::assert_debug),
                    FormatTrait::LowerExp => (sym::fmt_lower_exp, sym::assert_lower_exp),
                    FormatTrait::UpperExp => (sym::fmt_upper_exp, sym::assert_upper_exp),
                    FormatTrait::Octal => (sym::fmt_octal, sym::assert_octal),
                    FormatTrait::Pointer => (sym::fmt_pointer, sym::assert_pointer),
                    FormatTrait::Binary => (sym::fmt_binary, sym::assert_binary),
                    FormatTrait::LowerHex => (sym::fmt_lower_hex, sym::assert_lower_hex),
                    FormatTrait::UpperHex => (sym::fmt_upper_hex, sym::assert_upper_hex),
                };

                let arg_index = placeholder.argument.index.unwrap();
                let arg = &arguments[arg_index];
                let arg_span = arg.expr.span.with_ctxt(macsp_ctxt);

                let placeholder_span = placeholder
                    .span
                    .map_or(arg_span, |placeholder_span| placeholder_span.with_ctxt(macsp_ctxt));

                let arg_span = if matches!(arg.kind, FormatArgumentKind::Captured(_)) {
                    placeholder_span
                } else {
                    arg_span
                };

                let args_ident_expr = ctx.expr_ident(arg_span, args_ident, args_hir_id);

                let arg = arena.alloc(ctx.expr(
                    arg_span,
                    hir::ExprKind::Field(
                        args_ident_expr,
                        Ident::new(sym::integer(arg_index), arg_span),
                    ),
                ));

                let argument_checker = arena.alloc(ctx.expr_lang_item_type_relative(
                    placeholder_span,
                    hir::LangItem::FormatArgumentChecker,
                    assert_method,
                ));

                let checked_arg =
                    ctx.expr_call(placeholder_span, argument_checker, slice::from_ref(arg));

                (method, None, Some(checked_arg))
            }
        };

        let generic_args = generic_arg.map(|body_expr| {
            let parent_def_id = ctx.current_hir_id_owner.def_id;
            let node_id = ctx.next_node_id();

            let def_id = ctx.create_def(
                parent_def_id,
                node_id,
                kw::Empty,
                hir::def::DefKind::AnonConst,
                macsp,
            );

            let hir_id = ctx.lower_node_id(node_id);
            let body = ctx.lower_body(|_| (&[], body_expr));
            let anno_const = hir::AnonConst { hir_id, def_id, body, span: macsp };

            &*arena.alloc(hir::GenericArgs {
                args: slice::from_ref(arena.alloc(hir::GenericArg::Const(arena.alloc(
                    hir::ConstArg {
                        hir_id: ctx.next_id(),
                        kind: hir::ConstArgKind::Anon(arena.alloc(anno_const)),
                    },
                )))),
                constraints: &[],
                parenthesized: hir::GenericArgsParentheses::No,
                span_ext: rustc_span::DUMMY_SP,
            })
        });

        let args = arg.map(slice::from_ref).unwrap_or_default();

        fmt_fn_expr = make_method_call_expr(ctx, macsp, fmt_fn_expr, method, generic_args, args);
    }

    fmt_fn_expr = make_method_call_expr(ctx, macsp, fmt_fn_expr, sym::build, None, &[]);

    // Generate:
    //     <core::fmt::rt::Argument>::from_fmt_fn(
    //         <core::fmt::rt::FmtFnBuilder>::new()
    //             …
    //             .fmt_debug(args.1)
    //             .offset_argument_ptr::<1>()
    //             .write_str()
    //             .offset_literal_ptr::<1>()
    //             .fmt_display(args.0)
    //             .write_str()
    //             .build(),
    //     )

    let fmt_fn_argument =
        ctx.expr_call_mut(macsp, argument_from_fmt_fn, slice::from_ref(fmt_fn_expr));

    // Generate literal array:
    //     ["…", "…", "…", …]

    let literal_array_elements =
        arena.alloc_from_iter(literal_data.into_iter().map(|literal| ctx.expr_str(macsp, literal)));

    let literal_array_expr =
        arena.alloc(ctx.expr(macsp, hir::ExprKind::Array(literal_array_elements)));

    // Generate:
    //     const { <core::fmt::rt::CompileTimeData>::new(["…", "…", "…", …]) }

    let compile_time_data = {
        let parent_def_id = ctx.current_hir_id_owner.def_id;
        let node_id = ctx.next_node_id();

        let def_id = ctx.create_def(
            parent_def_id,
            node_id,
            kw::Empty,
            hir::def::DefKind::InlineConst,
            macsp,
        );

        let hir_id = ctx.lower_node_id(node_id);

        let body = ctx.lower_body(|ctx| {
            let compile_time_data_new = arena.alloc(ctx.expr_lang_item_type_relative(
                macsp,
                hir::LangItem::FormatCompileTimeData,
                sym::new,
            ));

            (
                &[],
                ctx.expr_call_mut(
                    macsp,
                    compile_time_data_new,
                    slice::from_ref(literal_array_expr),
                ),
            )
        });

        arena.alloc(
            ctx.expr(macsp, hir::ExprKind::ConstBlock(hir::ConstBlock { hir_id, def_id, body })),
        )
    };

    // Generate:
    //     <core::fmt::rt::Argument>::from_compile_time_data(
    //         &const { <core::fmt::rt::CompileTimeData>::new(["…", "…", "…", …]) },
    //     )

    let argument_from_compile_time_data = arena.alloc(ctx.expr_lang_item_type_relative(
        macsp,
        hir::LangItem::FormatArgument,
        sym::from_compile_time_data,
    ));

    let ref_compile_time_data = ctx.expr(
        macsp,
        hir::ExprKind::AddrOf(hir::BorrowKind::Ref, hir::Mutability::Not, compile_time_data),
    );

    let compile_time_data_argument = ctx.expr_call_mut(
        macsp,
        argument_from_compile_time_data,
        slice::from_ref(arena.alloc(ref_compile_time_data)),
    );

    // Generate:
    //     [
    //         <core::fmt::rt::Argument>::from_fmt_fn(
    //             <core::fmt::rt::FmtFnBuilder>::new()
    //                 …
    //                 .fmt_debug(args.1)
    //                 .offset_argument_ptr::<1>()
    //                 .write_str()
    //                 .offset_literal_ptr::<1>()
    //                 .fmt_display(args.0)
    //                 .write_str()
    //                 .build(),
    //         ),
    //         <core::fmt::rt::Argument>::from_compile_time_data(
    //             &const { <core::fmt::rt::CompileTimeData>::new(["…", "…", "…", …]) },
    //         ),
    //         <core::fmt::rt::Argument>::from_ref(<core::fmt::rt::Argument>::assert_…(args.0)),
    //         <core::fmt::rt::Argument>::from_ref(<core::fmt::rt::Argument>::assert_…(args.1)),
    //         <core::fmt::rt::Argument>::from_ref(<core::fmt::rt::Argument>::assert_…(args.2)),
    //         …
    //     ]

    let argument_data_elements =
        arena.alloc_from_iter([fmt_fn_argument, compile_time_data_argument].into_iter().chain(
            argument_data.into_iter().map(|(arg_index, arg_usage)| {
                let argument = &arguments[arg_index];
                let macsp_ctxt = macsp.ctxt();
                let arg_span = argument.expr.span.with_ctxt(macsp_ctxt);

                let (method, placeholder_span, arg_span) = match arg_usage {
                    ArgumentUsage::Ref => (sym::from_ref, arg_span, arg_span),
                    ArgumentUsage::Usize(placeholder_span) => {
                        if let Some(placeholder_span) = placeholder_span {
                            let placeholder_span = placeholder_span.with_ctxt(macsp_ctxt);

                            let arg_span =
                                if matches!(argument.kind, FormatArgumentKind::Captured(..)) {
                                    placeholder_span
                                } else {
                                    arg_span
                                };

                            (sym::from_usize, placeholder_span, arg_span)
                        } else {
                            (sym::from_usize, arg_span, arg_span)
                        }
                    }
                };

                let method_expr = arena.alloc(ctx.expr_lang_item_type_relative(
                    placeholder_span,
                    hir::LangItem::FormatArgument,
                    method,
                ));

                let args_ident_expr = ctx.expr_ident(arg_span, args_ident, args_hir_id);

                let args = slice::from_ref(arena.alloc(ctx.expr(
                    arg_span,
                    hir::ExprKind::Field(
                        args_ident_expr,
                        Ident::new(sym::integer(arg_index), macsp),
                    ),
                )));

                ctx.expr_call_mut(placeholder_span, method_expr, args)
            }),
        ));

    let argument_data_expr =
        arena.alloc(ctx.expr(macsp, hir::ExprKind::Array(argument_data_elements)));

    // Generate:
    //     args => [
    //         <core::fmt::rt::Argument>::from_fmt_fn(
    //             <core::fmt::rt::FmtFnBuilder>::new()
    //                 …
    //                 .fmt_debug(args.1)
    //                 .offset_argument_ptr::<1>()
    //                 .write_str()
    //                 .offset_literal_ptr::<1>()
    //                 .fmt_display(args.0)
    //                 .write_str()
    //                 .build(),
    //         ),
    //         <core::fmt::rt::Argument>::from_compile_time_data(
    //             &const { <core::fmt::rt::CompileTimeData>::new(["…", "…", "…", …]) },
    //         ),
    //         <core::fmt::rt::Argument>::from_ref(<core::fmt::rt::Argument>::assert_…(args.0)),
    //         <core::fmt::rt::Argument>::from_ref(<core::fmt::rt::Argument>::assert_…(args.1)),
    //         <core::fmt::rt::Argument>::from_ref(<core::fmt::rt::Argument>::assert_…(args.2)),
    //         …
    //     ],

    let match_arm = arena.alloc(ctx.arm(args_pat, argument_data_expr));

    // Generate:
    //     &match (&arg0, &arg1, &arg2, …) {
    //         args => [
    //             <core::fmt::rt::Argument>::from_fmt_fn(
    //                 <core::fmt::rt::FmtFnBuilder>::new()
    //                     …
    //                     .fmt_debug(args.1)
    //                     .offset_argument_ptr::<1>()
    //                     .write_str()
    //                     .offset_literal_ptr::<1>()
    //                     .fmt_display(args.0)
    //                     .write_str()
    //                     .build(),
    //             ),
    //             <core::fmt::rt::Argument>::from_compile_time_data(
    //                 &const { <core::fmt::rt::CompileTimeData>::new(["…", "…", "…", …]) },
    //             ),
    //             <core::fmt::rt::Argument>::from_ref(<core::fmt::rt::Argument>::assert_…(args.0)),
    //             <core::fmt::rt::Argument>::from_ref(<core::fmt::rt::Argument>::assert_…(args.1)),
    //             <core::fmt::rt::Argument>::from_ref(<core::fmt::rt::Argument>::assert_…(args.2)),
    //             …
    //         ],
    //     }

    let match_expr = arena.alloc(ctx.expr(
        macsp,
        hir::ExprKind::Match(args_tuple, slice::from_ref(match_arm), hir::MatchSource::FormatArgs),
    ));

    let ref_match_expr = ctx
        .expr(macsp, hir::ExprKind::AddrOf(hir::BorrowKind::Ref, hir::Mutability::Not, match_expr));

    // Generate estimated capacity value.

    let estimated_capacity = literal_length
        .and_then(|literal_length| {
            if !starts_with_literal && literal_length < 16 {
                // If the format string starts with an argument,
                // don't preallocate anything, unless length
                // of pieces is significant.
                Some(0)
            } else {
                // There are some arguments, so any additional push
                // will reallocate the string. To avoid that,
                // we're "pre-doubling" the capacity here.
                literal_length.checked_mul(2).filter(|&length| {
                    let target_pointer_width = ctx.tcx.sess.target.pointer_width;
                    let isize_max = (1_u64 << (target_pointer_width - 1)) - 1;

                    length <= isize_max // Make sure the capacity do not exceed `isize::MAX`.
                })
            }
        })
        .unwrap_or(0);

    let estimated_capacity_expr = ctx.expr(
        macsp,
        hir::ExprKind::Lit(arena.alloc(hir::Lit {
            span: macsp,
            node: ast::LitKind::Int(
                u128::from(estimated_capacity).into(),
                ast::LitIntType::Unsigned(ast::UintTy::Usize),
            ),
        })),
    );

    // Generate:
    //     unsafe { ::core::fmt::rt::UnsafeArg::new() }

    let unsafe_arg_new = arena.alloc(ctx.expr_lang_item_type_relative(
        macsp,
        hir::LangItem::FormatUnsafeArg,
        sym::new,
    ));

    let unsafe_arg_new_call_expr = ctx.expr_call(macsp, unsafe_arg_new, &[]);
    let unsafe_arg_expr_hir_id = ctx.next_id();

    let unsafe_arg = ctx.expr_block(arena.alloc(hir::Block {
        stmts: &[],
        expr: Some(unsafe_arg_new_call_expr),
        hir_id: unsafe_arg_expr_hir_id,
        rules: hir::BlockCheckMode::UnsafeBlock(hir::UnsafeSource::CompilerGenerated),
        span: macsp,
        targeted_by_break: false,
    }));

    // Generate:
    //     <core::fmt::Arguments>::new(
    //         &match (&arg0, &arg1, &arg2, …) {
    //             args => [
    //                 <core::fmt::rt::Argument>::from_fmt_fn(
    //                     <core::fmt::rt::FmtFnBuilder>::new()
    //                         …
    //                         .fmt_debug(args.1)
    //                         .offset_argument_ptr::<1>()
    //                         .write_str()
    //                         .offset_literal_ptr::<1>()
    //                         .fmt_display(args.0)
    //                         .write_str()
    //                         .build(),
    //                 ),
    //                 <core::fmt::rt::Argument>::from_compile_time_data(
    //                     &const { <core::fmt::rt::CompileTimeData>::new(["…", "…", "…", …]) },
    //                 ),
    //                 <core::fmt::rt::Argument>::from_ref(<core::fmt::rt::Argument>::assert_…(args.0)),
    //                 <core::fmt::rt::Argument>::from_ref(<core::fmt::rt::Argument>::assert_…(args.1)),
    //                 <core::fmt::rt::Argument>::from_ref(<core::fmt::rt::Argument>::assert_…(args.2)),
    //                 …
    //             ],
    //         },
    //         estimated_capacity,
    //         unsafe { ::core::fmt::rt::UnsafeArg::new() }
    //     )

    let arguments_new = arena.alloc(ctx.expr_lang_item_type_relative(
        macsp,
        hir::LangItem::FormatArguments,
        sym::new,
    ));

    let arguments_new_call = hir::ExprKind::Call(
        arguments_new,
        arena.alloc([ref_match_expr, estimated_capacity_expr, unsafe_arg]),
    );

    arguments_new_call
}

fn for_all_argument_indexes(template: &mut [FormatArgsPiece], mut f: impl FnMut(&mut usize)) {
    for piece in template {
        let FormatArgsPiece::Placeholder(placeholder) = piece else { continue };
        if let Ok(index) = &mut placeholder.argument.index {
            f(index);
        }
        if let Some(FormatCount::Argument(FormatArgPosition { index: Ok(index), .. })) =
            &mut placeholder.format_options.width
        {
            f(index);
        }
        if let Some(FormatCount::Argument(FormatArgPosition { index: Ok(index), .. })) =
            &mut placeholder.format_options.precision
        {
            f(index);
        }
    }
}
