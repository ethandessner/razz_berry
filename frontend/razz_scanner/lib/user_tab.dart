import 'package:flutter/material.dart';

class UserTab extends StatelessWidget {
  const UserTab({super.key});
  @override
  Widget build(BuildContext context) {
    return Center(child: Text('User Tab', style: Theme.of(context).textTheme.headlineMedium));
  }
}
